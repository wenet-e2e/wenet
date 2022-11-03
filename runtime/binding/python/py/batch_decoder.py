# Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
#               2022 SoundDataConverge Co.LTD (Weiliang Chong)
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

from typing import List, Optional

import _wenet

from .hub import Hub


class BatchDecoder:

    def __init__(self,
                 model_dir: Optional[str] = None,
                 lang: str = 'chs',
                 nbest: int = 1,
                 enable_timestamp: bool = False,
                 context: Optional[List[str]] = None,
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
        if model_dir is None:
            model_dir = Hub.get_model_by_lang(lang)

        self.d = _wenet.BatchRecognizer(model_dir)

        self.set_language(lang)
        self.enable_timestamp(enable_timestamp)
        if context is not None:
            self.add_context(context)
            self.set_context_score(context_score)

    def __del__(self):
        del self.d

    def enable_timestamp(self, flag: bool):
        tag = 1 if flag else 0
        self.d.set_enable_timestamp(tag)

    def add_context(self, contexts: List[str]):
        for c in contexts:
            assert isinstance(c, str)
            self.d.AddContext(c)

    def set_context_score(self, score: float):
        self.d.set_context_score(score)

    def set_language(self, lang: str):
        assert lang in ['chs', 'en']
        self.d.set_language(lang)

    def decode(self, pcms: List[bytes]) -> str:
        """ Decode the input data

        Args:
            pcms: a list of wav pcm
        """
        assert isinstance(pcms[0], bytes)
        result = self.d.Decode(pcms)
        return result
