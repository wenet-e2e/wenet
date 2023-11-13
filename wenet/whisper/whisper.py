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

import base64
import gzip

import numpy as np
import torch

from wenet.transformer.asr_model import ASRModel
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
        self.n_vocab = vocab_size
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            len(decoder.decoders), decoder.decoders[0].self_attn.h, dtype=torch.bool
        )
        all_heads[len(decoder.decoders) // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            len(decoder.decoders), decoder.decoders[0].self_attn.h
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.jit.export
    @property
    def is_multilingual(self):
        return self.n_vocab >= 51865

    @torch.jit.export
    @property
    def num_languages(self):
        return self.n_vocab - 51765 - int(self.is_multilingual)
