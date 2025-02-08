# Copyright (c) 2025 Wenet Community. authors: Mddct(Dinghao Zhou)
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

from typing import Optional, Tuple

import torch
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.utils.common import IGNORE_ID


class FireReadModel(ASRModel):

    def __init__(
        self,
        vocab_size: int,
        encoder: ConformerEncoder,
        decoder: TransformerDecoder,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
    ):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens)
        assert reverse_weight == 0.0
        assert special_tokens is not None
        self.sos = special_tokens["sos"]
        self.eos = special_tokens["eos"]
        self.decode_maxlen = self.decoder.embed[1].max_len

    @torch.jit.unused
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('FiredASR don\'t support streaming')
