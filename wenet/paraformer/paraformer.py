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

# NOTE: This file is only for loading ali-paraformer-large-model and inference

from typing import Dict, List, Optional, Tuple

import torch

from wenet.cif.predictor import Predictor
from wenet.paraformer.layers import SanmDecoer, SanmEncoder
from wenet.paraformer.layers import LFR
from wenet.paraformer.search import (paraformer_beam_search,
                                     paraformer_greedy_search)
from wenet.transformer.search import DecodeResult
from wenet.utils.mask import make_pad_mask


class Paraformer(torch.nn.Module):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf

    """

    def __init__(self, encoder: SanmEncoder, decoder: SanmDecoer,
                 predictor: Predictor):

        super().__init__()
        self.lfr = LFR()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

        self.sos = 1
        self.eos = 2

    @torch.jit.ignore(drop=True)
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
        raise NotImplementedError("Training is currently not supported")

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
