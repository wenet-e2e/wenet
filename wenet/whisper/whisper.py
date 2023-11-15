# Copyright (c) 2023 Wenet Community. (authors: Xingchen Song)
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
#
# Modified from [Whisper](https://github.com/openai/whisper)

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.utils.common import IGNORE_ID


class Whisper(ASRModel):
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight, ignore_id,
                         reverse_weight, lsm_weight, length_normalized_loss)
        # FIXME(xcsong): rewrite sos & eos
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

    # TODO(xcsong): time align
    def set_alignment_heads(self, dump: bytes):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.n_vocab - 51765 - int(self.is_multilingual)
