# Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
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

import sys
from typing import List, Optional, Union

import librosa
import numpy as np
# import torch to avoid libtorch.so not found error
import torch  # noqa

import _wenet

from wenetruntime.hub import Hub


class Decoder:

    def __init__(self,
                 model_dir: Optional[str] = None,
                 lang: str = 'chs',
                 nbest: int = 1,
                 enable_timestamp: bool = False,
                 context: Optional[List[str]] = None,
                 context_score: float = 3.0,
                 continuous_decoding: bool = False,
                 streaming: bool = False):
        """ Init WeNet decoder
        Args:
            lang: language type of the model
            nbest: nbest number for the final result
            enable_timestamp: whether to enable word level timestamp
               for the final result
            context: context words
            context_score: bonus score when the context is matched
            continuous_decoding: enable countinous decoding or not
            streaming: streaming mode
        """
        if model_dir is None:
            model_dir = Hub.get_model_by_lang(lang)

        self.d = _wenet.wenet_init(model_dir)

        self.set_language(lang)
        self.set_nbest(nbest)
        self.enable_timestamp(enable_timestamp)
        if context is not None:
            self.add_context(context)
            self.set_context_score(context_score)
        self.set_continuous_decoding(continuous_decoding)
        chunk_size = 16 if streaming else -1
        self.set_chunk_size(chunk_size)

    def __del__(self):
        _wenet.wenet_free(self.d)

    def reset(self):
        """ Reset status for next decoding """
        _wenet.wenet_reset(self.d)

    def set_nbest(self, n: int):
        assert n >= 1
        assert n <= 10
        _wenet.wenet_set_nbest(self.d, n)

    def enable_timestamp(self, flag: bool):
        tag = 1 if flag else 0
        _wenet.wenet_set_timestamp(self.d, tag)

    def add_context(self, contexts: List[str]):
        for c in contexts:
            assert isinstance(c, str)
            _wenet.wenet_add_context(self.d, c)

    def set_context_score(self, score: float):
        _wenet.wenet_set_context_score(self.d, score)

    def set_language(self, lang: str):
        assert lang in ['chs', 'en']
        _wenet.wenet_set_language(self.d, lang)

    def set_continuous_decoding(self, continuous_decoding: bool):
        flag = 1 if continuous_decoding else 0
        _wenet.wenet_set_continuous_decoding(self.d, flag)

    def set_chunk_size(self, chunk_size: int):
        _wenet.wenet_set_chunk_size(self.d, chunk_size)

    def decode(self,
               audio: Union[str, bytes, np.ndarray],
               last: bool = True) -> str:
        """ Decode the input audio

        Args:
            audio: string, bytes, or np.ndarray
            last: if it is the last package of the data, only for streaming
        """
        if isinstance(audio, str):
            data, _ = librosa.load(audio, sr=16000)
            data = data * (1 << 15)
            data = data.astype(np.int16).tobytes()
            finish = 1
        elif isinstance(audio, np.ndarray):
            finish = 1 if last else 0
            if audio.max() < 1:  # the audio is normalized
                data = data * (1 << 15)
            data = data.astype(np.int16).tobytes()
        elif isinstance(audio, bytes):
            finish = 1 if last else 0
            data = audio
        else:
            print('Unsupport audio type {}'.format(type(audio)))
            sys.exit(-1)
        result = _wenet.wenet_decode(self.d, data, len(data), finish)
        if last:  # Reset status for next decoding automatically
            self.reset()
        return result

    def decode_wav(self, wav_file: str) -> str:
        """ Deprecated, will remove soon """
        return self.decode(wav_file)
