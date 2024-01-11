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
from wenet.paraformer.cif import Cif, cif_without_hidden

from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.layers import LFR
from wenet.paraformer.search import (paraformer_beam_search,
                                     paraformer_greedy_search)
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.search import (DecodeResult, ctc_greedy_search,
                                      ctc_prefix_beam_search)
from wenet.utils.common import IGNORE_ID, add_sos_eos, th_accuracy
from wenet.utils.mask import make_non_pad_mask


class Predictor(torch.nn.Module):

    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0.0,
        tail_threshold=0.45,
        residual=True,
        cnn_groups=0,
        smooth_factor2=0.25,
        noise_threshold2=0.01,
        upsample_times=3,
    ):
        super().__init__()
        self.predictor = Cif(idim, l_order, r_order, threshold, dropout,
                             smooth_factor, noise_threshold, tail_threshold,
                             residual, cnn_groups)

        # accurate timestamp branch
        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold
        self.upsample_times = upsample_times
        self.noise_threshold2 = noise_threshold2
        self.tp_upsample_cnn = torch.nn.ConvTranspose1d(
            idim, idim, self.upsample_times, self.upsample_times)
        self.tp_blstm = torch.nn.LSTM(idim,
                                      idim,
                                      1,
                                      bias=True,
                                      batch_first=True,
                                      dropout=0.0,
                                      bidirectional=True)
        self.tp_output = torch.nn.Linear(idim * 2, 1)

    def forward(self,
                hidden,
                target_label: Optional[torch.Tensor] = None,
                mask: torch.Tensor = torch.tensor(0),
                ignore_id: int = -1,
                mask_chunk_predictor: Optional[torch.Tensor] = None,
                target_label_length: Optional[torch.Tensor] = None):

        acoustic_embeds, token_num, alphas, cif_peak = self.predictor(
            hidden, target_label, mask, ignore_id, mask_chunk_predictor,
            target_label_length)

        output, (_, _) = self.tp_blstm(
            self.tp_upsample_cnn(hidden.transpose(1, 2)).transpose(1, 2))
        tp_alphas = torch.sigmoid(self.tp_output(output))
        tp_alphas = torch.nn.functional.relu(tp_alphas * self.smooth_factor2 -
                                             self.noise_threshold2)

        mask = mask.repeat(1, self.upsample_times,
                           1).transpose(-1,
                                        -2).reshape(tp_alphas.shape[0], -1)
        mask = mask.unsqueeze(-1)
        tp_alphas = tp_alphas * mask
        tp_alphas = tp_alphas.squeeze(-1)
        tp_token_num = tp_alphas.sum(-1)

        return acoustic_embeds, token_num, alphas, cif_peak, tp_alphas, tp_token_num


class Paraformer(ASRModel):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf

    """

    def __init__(self,
                 vocab_size: int,
                 encoder: BaseEncoder,
                 decoder: TransformerDecoder,
                 predictor: Predictor,
                 ctc: CTC,
                 ctc_weight: float = 0.5,
                 ignore_id: int = -1,
                 lsm_weight: float = 0,
                 length_normalized_loss: bool = False,
                 sampler: bool = True,
                 sampling_ratio: float = 0.75,
                 add_eos: bool = True,
                 special_tokens: Optional[Dict] = None,
                 apply_non_blank_embedding: bool = False):
        assert isinstance(encoder,
                          SanmEncoder), isinstance(decoder, SanmDecoder)
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         IGNORE_ID, 0.0, lsm_weight, length_normalized_loss,
                         None, apply_non_blank_embedding)
        if ctc_weight == 0.0:
            del ctc
        self.predictor = predictor
        self.lfr = LFR()

        assert special_tokens is not None
        self.sos = special_tokens['<sos>']
        self.eos = special_tokens['<eos>']

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
        """Frontend + Encoder + Predictor + Decoder + Calc loss
        """
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        # 0 encoder
        encoder_out, encoder_out_mask = self._forward_encoder(
            speech, speech_lengths)

        # 1 predictor
        ys_pad, ys_pad_lens = text, text_lengths
        if self.add_eos:
            _, ys_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = text_lengths + 1
        acoustic_embd, token_num, _, _, _, tp_token_num = self.predictor(
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
        loss_quantity_tp = torch.nn.functional.l1_loss(
            tp_token_num, ys_pad_lens.to(token_num.dtype),
            reduction='sum') / ys_pad_lens.sum().to(token_num.dtype)

        loss_decoder, acc_att = self._calc_att_loss(encoder_out,
                                                    encoder_out_mask, ys_pad,
                                                    acoustic_embd, ys_pad_lens)
        loss = loss_decoder
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc
        loss = loss + loss_quantity + loss_quantity_tp
        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_decoder": loss_decoder,
            "loss_quantity": loss_quantity,
            "loss_quantity_tp": loss_quantity_tp,
            "th_accuracy": acc_att,
        }

    def _calc_att_loss(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor, ys_pad_emb: torch.Tensor,
            ys_pad_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_out, _, _ = self.decoder(encoder_out, encoder_mask, ys_pad_emb,
                                         ys_pad_lens)
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),
                              ys_pad,
                              ignore_label=self.ignore_id)
        return loss_att, acc_att

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

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO(Mddct): support chunk by chunk
        assert simulate_streaming is False
        features, features_lens = self.lfr(speech, speech_lengths)
        features_lens = features_lens.to(speech_lengths.dtype)
        encoder_out, encoder_out_mask = self.encoder(features, features_lens,
                                                     decoding_chunk_size,
                                                     num_decoding_left_chunks)
        return encoder_out, encoder_out_mask

    @torch.jit.export
    def forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self._forward_paraformer(speech, speech_lengths)
        return res['decoder_out'], res['decoder_out_lens'], res['tp_alphas']

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(Mddct): fix
        xs_lens = torch.tensor(xs.size(1), dtype=torch.int)
        encoder_out, _ = self._forward_encoder(xs, xs_lens)
        return encoder_out, att_cache, cnn_cache

    @torch.jit.export
    def forward_cif_peaks(self, alphas: torch.Tensor,
                          token_nums: torch.Tensor) -> torch.Tensor:
        cif2_token_nums = alphas.sum(-1)
        scale_alphas = alphas / (cif2_token_nums / token_nums).unsqueeze(1)
        peaks = cif_without_hidden(scale_alphas,
                                   self.predictor.predictor.threshold - 1e-4)

        return peaks

    def _forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
    ) -> Dict[str, torch.Tensor]:
        # encoder
        encoder_out, encoder_out_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks)

        # cif predictor
        acoustic_embed, token_num, _, _, tp_alphas, _ = self.predictor(
            encoder_out, mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lengths.dtype)

        # decoder
        decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
                                         acoustic_embed, token_num)
        decoder_out = decoder_out.log_softmax(dim=-1)

        return {
            "encoder_out": encoder_out,
            "encoder_out_mask": encoder_out_mask,
            "decoder_out": decoder_out,
            "tp_alphas": tp_alphas,
            "decoder_out_lens": token_num
        }

    def decode(self,
               methods: List[str],
               speech: torch.Tensor,
               speech_lengths: torch.Tensor,
               beam_size: int,
               decoding_chunk_size: int = -1,
               num_decoding_left_chunks: int = -1,
               ctc_weight: float = 0,
               simulate_streaming: bool = False,
               reverse_weight: float = 0,
               context_graph=None,
               blank_id: int = 0,
               blank_penalty: float = 0.0) -> Dict[str, List[DecodeResult]]:
        res = self._forward_paraformer(speech, speech_lengths,
                                       decoding_chunk_size,
                                       num_decoding_left_chunks)
        encoder_out, encoder_mask, decoder_out, decoder_out_lens, tp_alphas = res[
            'encoder_out'], res['encoder_out_mask'], res['decoder_out'], res[
                'decoder_out_lens'], res['tp_alphas']
        peaks = self.forward_cif_peaks(tp_alphas, decoder_out_lens)
        results = {}
        if 'paraformer_greedy_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_greedy_result = paraformer_greedy_search(
                decoder_out, decoder_out_lens, peaks)
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
        if 'ctc_greedy_search' in methods or 'ctc_prefix_beam_search' in methods:
            ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
            encoder_lens = encoder_mask.squeeze(1).sum(1)
            if 'ctc_greedy_search' in methods:
                results['ctc_greedy_search'] = ctc_greedy_search(
                    ctc_probs, encoder_lens, blank_id)
            if 'ctc_prefix_beam_search' in methods:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
                results['ctc_prefix_beam_search'] = ctc_prefix_result
        return results
