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

from typing import List

import _wenet


class Decoder:
    def __init__(self,
                 model_dir: str,
                 lang: str = 'chs',
                 nbest: int = 1,
                 enable_timestamp: bool = False,
                 context: List[str] = None,
                 context_score: float = 3.0):
        """ Init WeNet decoder
        Args:
            lang: language type of the model
            nbest: nbest number for the final result
            enable_timestamp: whether to enable word level timestamp
               for the final result
            context: context words
            context_score: bonus score when the context is matched
        """
        self.d = _wenet.wenet_init(model_dir)
        self.set_language(lang)
        self.set_nbest(nbest)
        self.enable_timestamp(enable_timestamp)
        if context is not None:
            self.add_context(context)
        self.set_context_score(context_score)

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

    def decode(self, pcm: bytes, last: bool = True) -> str:
        """ Decode the input data

        Args:
            pcm: wav pcm
            last: if it is the last package of the data
        """
        assert isinstance(pcm, bytes)
        finish = 1 if last else 0
        _wenet.wenet_decode(self.d, pcm, len(pcm), finish)
        result = _wenet.wenet_get_result(self.d)
        return result

    def decode_wav(self, wav_file: str) -> str:
        """ Decode wav file, we only support:
            1. 16k sample rate
            2. mono channel
            3. sample widths is 16 bits / 2 bytes
        """
        import wave
        with wave.open(wav_file, 'rb') as fin:
            assert fin.getnchannels() == 1
            assert fin.getsampwidth() == 2
            assert fin.getframerate() == 16000
            wav = fin.readframes(fin.getnframes())
        return self.decode(wav, True)
