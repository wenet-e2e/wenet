
# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)
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

import torch

from wenet.models.transformer.asr_model import ASRModel
from wenet.models.transformer.ctc import CTC
from wenet.models.transformer.decoder import TransformerDecoder
from wenet.models.transformer.encoder import TransformerEncoder
from wenet.utils.common import IGNORE_ID


class Qwen3OmniAUT(ASRModel):

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder = torch.nn.Identity(),
        ctc: CTC = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: dict = None,
    ):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens)
        assert reverse_weight == 0.0

    def tie_or_clone_weights(self, jit_mode: bool = True):
        # dummy method
        pass
