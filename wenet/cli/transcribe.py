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

import argparse
import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.cli.hub import Hub
from wenet.utils.common import remove_duplicates_and_blank
from wenet.utils.file_utils import read_symbol_table


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('audio_file', help='audio file to transcribe')
    parser.add_argument('--language',
                        choices=[
                            'chinese',
                            'english',
                        ],
                        default='chinese',
                        help='language type')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_dir = Hub.get_model_by_lang(args.language)
    model_path = os.path.join(model_dir, 'final.zip')
    units_path = os.path.join(model_dir, 'units.txt')
    model = torch.jit.load(model_path)
    symbol_table = read_symbol_table(units_path)
    char_dict = {v: k for k, v in symbol_table.items()}
    waveform, sample_rate = torchaudio.load(args.audio_file, normalize=False)
    waveform = waveform.to(torch.float)
    feats = kaldi.fbank(waveform,
                        num_mel_bins=80,
                        frame_length=25,
                        frame_shift=10,
                        energy_floor=0.0,
                        sample_frequency=16000)
    feats = feats.unsqueeze(0)
    encoder_out, _, _ = model.forward_encoder_chunk(feats, 0, -1)
    ctc_probs = model.ctc_activation(encoder_out)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)
    topk_index = topk_index.squeeze().tolist()
    hyp = remove_duplicates_and_blank(topk_index)
    hyp = [char_dict[x] for x in hyp]
    result = ''.join(hyp)
    print(result)


if __name__ == "__main__":
    main()
