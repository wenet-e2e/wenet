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

from wenet.cif.predictor import MAELoss
from wenet.paraformer.search import paraformer_beam_search, paraformer_greedy_search
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.search import (DecodeResult, ctc_greedy_search,
                                      ctc_prefix_beam_search)
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy)
from wenet.utils.mask import make_pad_mask


class Paraformer(ASRModel):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        predictor,
        ctc_weight: float = 0.5,
        predictor_weight: float = 1.0,
        predictor_bias: int = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= predictor_weight <= 1.0, predictor_weight

        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss)
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.criterion_pre = MAELoss(normalize_length=length_normalized_loss)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, loss_pre = self._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths)
        else:
            # loss_att = None
            # loss_pre = None
            loss_att: torch.Tensor = torch.tensor(0)
            loss_pre: torch.Tensor = torch.tensor(0)

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att + self.predictor_weight * loss_pre
        # elif loss_att is None:
        elif loss_att == torch.tensor(0):
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + \
                (1 - self.ctc_weight) * loss_att + \
                self.predictor_weight * loss_pre
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "loss_pre": loss_pre
        }

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = \
            self.predictor(encoder_out, ys_pad, encoder_mask,
                           ignore_id=self.ignore_id)
        # 1. Forward decoder
        decoder_out, _, _ = self.decoder(encoder_out, encoder_mask,
                                         pre_acoustic_embeds, ys_pad_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre: torch.Tensor = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length)

        return loss_att, acc_att, loss_pre

    def calc_predictor(self, encoder_out, encoder_out_lens):
        encoder_mask = (~make_pad_mask(
            encoder_out_lens, max_len=encoder_out.size(1))[:, None, :]).to(
                encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = \
            self.predictor(
                encoder_out, None,
                encoder_mask,
                ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(self, encoder_out, encoder_mask,
                                   sematic_embeds, ys_pad_lens):
        decoder_out, _, _ = self.decoder(encoder_out, encoder_mask,
                                         sematic_embeds, ys_pad_lens)
        return decoder_out, ys_pad_lens

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
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        results = {}

        ctc_probs: Optional[torch.Tensor] = None
        if 'ctc_greedy_search' in methods:
            ctc_probs = self.ctc.log_softmax(encoder_out)
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens)
        if 'ctc_prefix_beam_search' in methods:
            if ctc_probs is None:
                ctc_probs = self.ctc.log_softmax(encoder_out)
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size)
            results['ctc_prefix_beam_search'] = ctc_prefix_result

        decoder_out: Optional[torch.Tensor] = None
        decoder_out_lens: Optional[torch.Tensor] = None
        # TODO(Mddct): add timestamp from predictor's alpha
        if ('paraformer_greedy_search' in methods
                or 'paraformer_beam_search' in methods):
            acoustic_embed, token_nums, _, _ = self.calc_predictor(
                encoder_out, encoder_lens)
            decoder_out, decoder_out_lens = self.cal_decoder_with_predictor(
                encoder_out, encoder_mask, acoustic_embed, token_nums)
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
                decoder_out, decoder_out_lens, beam_size=beam_size)
            results['paraformer_beam_search'] = paraformer_beam_result

        return results
