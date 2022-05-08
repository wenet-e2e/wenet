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

import _wenet


class Decoder:
    def __init__(self, model_dir: str):
        self.d = _wenet.wenet_init(model_dir)

    def __del__(self):
        _wenet.wenet_free(self.d)

    def reset(self):
        """ Reset status for next decoding """
        _wenet.wenet_reset(self.d)

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
