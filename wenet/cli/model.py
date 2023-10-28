# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.cli.hub import Hub
from wenet.utils.ctc_utils import gen_timestamps_from_peak
from wenet.utils.file_utils import read_symbol_table
from wenet.transformer.search import (attention_rescoring,
                                      ctc_prefix_beam_search)


class Model:
    def __init__(self, model_dir: str):
        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        symbol_table = read_symbol_table(units_path)
        self.char_dict = {v: k for k, v in symbol_table.items()}

    def transcribe(self, audio_file: str, tokens_info: bool = False):
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=16000)
        feats = feats.unsqueeze(0)
        encoder_out, _, _ = self.model.forward_encoder_chunk(feats, 0, -1)
        encoder_lens = torch.tensor([encoder_out.size(1)], dtype=torch.long)
        ctc_probs = self.model.ctc_activation(encoder_out)
        ctc_prefix_results = ctc_prefix_beam_search(ctc_probs, encoder_lens, 2)
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence

        if tokens_info:
            frame_rate = self.model.subsampling_rate(
            ) * 0.01  # 0.01 seconds per frame
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(res.times, max_duration,
                                             frame_rate, 1.0)
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.char_dict[x],
                    'start': times[i][0],
                    'end': times[i][1],
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info
        return result


def load_model(language: str = None, model_dir: str = None) -> Model:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang(language)
    return Model(model_dir)
