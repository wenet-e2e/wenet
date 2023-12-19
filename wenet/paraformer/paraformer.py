# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet) and
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from typing import Dict, List, Optional, Tuple

import torch
from wenet.paraformer.cif import Cif

from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.layers import LFR
from wenet.paraformer.search import (paraformer_beam_search,
                                     paraformer_greedy_search)
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import DecodeResult
from wenet.utils.common import add_sos_eos
from wenet.utils.mask import make_non_pad_mask


class Paraformer(torch.nn.Module):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf

    """

    def __init__(self,
                 vocab_size: int,
                 encoder: BaseEncoder,
                 decoder: TransformerDecoder,
                 predictor: Cif,
                 ctc: Optional[CTC] = None,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0,
                 length_normalized_loss: bool = False,
                 sampler: bool = True,
                 sampling_ratio: float = 0.75,
                 add_eos: bool = True,
                 special_tokens: Optional[Dict] = None,
                 **kwargs):
        assert isinstance(encoder,
                          SanmEncoder), isinstance(decoder, SanmDecoder)
        super().__init__()
        self.ctc_weight = ctc_weight
        self.ctc = ctc
        if ctc_weight == 0.0:
            del ctc
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

        self.lfr = LFR()

        assert special_tokens is not None
        self.sos = special_tokens['<sos>']
        self.eos = special_tokens['<eos>']
        self.ignore_id = ignore_id

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss)

        self.sampler = sampler
        self.sampling_ratio = sampling_ratio
        if sampler:
            self.embed = self.decoder.embed
        else:
            del self.decoder.embed
        # NOTE(Mddct): add eos in tail of labels for predictor
        # eg:
        #    gt:         你 好 we@@ net
        #    labels:     你 好 we@@ net eos
        self.add_eos = add_eos

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        batch: Dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss
        """
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        features, features_lens = self.lfr(speech, speech_lengths)
        features_lens = features_lens.to(speech_lengths.dtype)

        # 0 encoder
        encoder_out, encoder_out_mask = self.encoder(features, features_lens)

        # 1 predictor
        ys_pad, ys_pad_lens = text, text_lengths
        if self.add_eos:
            _, ys_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = text_lengths + 1
        acoustic_embd, token_num, _, _ = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, self.ignore_id)

        # 2 decoder with sampler
        # TODO(Mddct): support mwer here
        acoustic_embd = self._sampler(
            encoder_out,
            encoder_out_mask,
            ys_pad,
            ys_pad_lens,
            acoustic_embd,
        )
        decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
                                         acoustic_embd, ys_pad_lens)

        # 3 loss
        # 3.1 ctc branhch
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0:
            loss_ctc, _ = self._forward_ctc(encoder_out, encoder_out_mask,
                                            text, text_lengths)
        # 3.2 quantity loss for cif
        loss_quantity = torch.nn.functional.l1_loss(
            token_num,
            ys_pad_lens.to(token_num.dtype),
            reduction='sum',
        )
        loss_quantity = loss_quantity / ys_pad_lens.sum().to(token_num.dtype)

        # TODO(Mddc): thu acc
        loss_decoder = self.criterion_att(decoder_out, ys_pad)
        loss = loss_decoder
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc
        loss = loss + loss_quantity
        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_decoder": loss_decoder,
            "loss_quantity": loss_quantity,
        }

    @torch.jit.ignore(drop=True)
    def _forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                       text_lengths)
        return loss_ctc, ctc_probs

    @torch.jit.ignore(drop=True)
    def _sampler(self, encoder_out, encoder_out_mask, ys_pad, ys_pad_lens,
                 pre_acoustic_embeds):
        device = encoder_out.device
        B, _ = ys_pad.size()

        tgt_mask = make_non_pad_mask(ys_pad_lens)
        # zero the ignore id
        ys_pad = ys_pad * tgt_mask
        ys_pad_embed = self.embed(ys_pad)  # [B, T, L]
        with torch.no_grad():
            decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
                                             pre_acoustic_embeds, ys_pad_lens)
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = tgt_mask
            same_num = ((pred_tokens == ys_pad) * nonpad_positions).sum(1)
            input_mask = torch.ones_like(
                nonpad_positions,
                device=device,
                dtype=tgt_mask.dtype,
            )
            for li in range(B):
                target_num = (ys_pad_lens[li] -
                              same_num[li].sum()).float() * self.sampling_ratio
                target_num = target_num.long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(ys_pad_lens[li],
                                             device=device)[:target_num],
                        value=0,
                    )
            input_mask = torch.where(input_mask > 0, 1, 0)
            input_mask = input_mask * tgt_mask
            input_mask_expand = input_mask.unsqueeze(2)  # [B, T, 1]

        sematic_embeds = torch.where(input_mask_expand == 1,
                                     pre_acoustic_embeds, ys_pad_embed)
        # zero out the paddings
        return sematic_embeds * tgt_mask.unsqueeze(2)

    @torch.jit.export
    def forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, features_lens = self.lfr(speech, speech_lengths)
        features_lens = features_lens.to(speech_lengths.dtype)
        # encoder
        encoder_out, encoder_out_mask = self.encoder(features, features_lens)

        # cif predictor
        acoustic_embed, token_num, _, _ = self.predictor(encoder_out,
                                                         mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lengths.dtype)

        # decoder
        decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
                                         acoustic_embed, token_num)
        decoder_out = decoder_out.log_softmax(dim=-1)
        return decoder_out, token_num

    def decode(self,
               methods: List[str],
               speech: torch.Tensor,
               speech_lengths: torch.Tensor,
               beam_size: int,
               decoding_chunk_size: int = -1,
               num_decoding_left_chunks: int = -1,
               ctc_weight: float = 0,
               simulate_streaming: bool = False,
               reverse_weight: float = 0) -> Dict[str, List[DecodeResult]]:
        decoder_out, decoder_out_lens = self.forward_paraformer(
            speech, speech_lengths)

        results = {}
        if 'paraformer_greedy_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_greedy_result = paraformer_greedy_search(
                decoder_out, decoder_out_lens)
            results['paraformer_greedy_search'] = paraformer_greedy_result
        if 'paraformer_beam_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_beam_result = paraformer_beam_search(
                decoder_out,
                decoder_out_lens,
                beam_size=beam_size,
                eos=self.eos)
            results['paraformer_beam_search'] = paraformer_beam_result

        return results
