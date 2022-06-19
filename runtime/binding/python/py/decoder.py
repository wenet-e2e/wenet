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

import os
import tarfile
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

import _wenet
import tqdm

# TODO(Mddct): make assets class to support other language
Assets = {
    # wenetspeech
    "chs":
    "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20220506_u2pp_conformer_libtorch.tar.gz",
    # gigaspeech
    "en":
    "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_libtorch.tar.gz"
}


class Decoder:

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
        if not model_dir:
            model_dir = self.get_model_by_lang(lang)

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

    def get_model_by_lang(self, lang: str):

        assert lang in ['chs', 'en']
        # NOTE(Mddct): model_dir structure
        # Path.Home()/.went
        # - chs
        #    - 20220506_u2pp_conformer_libtorch
        #       - words.txt
        #       - ....
        # - en
        model_dir_parent = os.path.join(Path.home(), ".wenet", lang)

        model_dir = Assets[lang].split("/")[-1].replace(".tar.gz", "")
        model_path = os.path.join(model_dir_parent, model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_dir_parent, exist_ok=True)
            return self.download_model(Assets[lang], model_dir)

        return model_path

    def download_model(self, model_url: str, model_dir: str):
        assert os.path.exists(model_url)

        def progress_hook(t):
            last_b = [0]

            def update_to(b=1, bsize=1, tsize=None):
                if tsize not in (None, -1):
                    t.total = tsize
                displayed = t.update((b - last_b[0]) * bsize)
                last_b[0] = b
                return displayed

            return update_to

        # *.tar.gz
        model_name = model_url.split("/")[-1]
        tar_model_path = os.path.join(model_dir, model_name)

        with tqdm.tqdm(unit='B',
                       unit_scale=True,
                       unit_divisor=1024,
                       miniters=1,
                       desc=(model_name)) as t:
            urlretrieve(model_url,
                        filename=tar_model_path,
                        reporthook=progress_hook(t),
                        data=None)
            t.total = t.n

            with tarfile.open(tar_model_path) as f:
                for name in f.getnames():
                    f.extract(name, path=model_dir)
        return os.path.join(model_dir, model_name).replace(".tar.gz", "", -1)

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
