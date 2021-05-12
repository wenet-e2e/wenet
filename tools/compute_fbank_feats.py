# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
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

import torchaudio
import torchaudio.compliance.kaldi as kaldi

import wenet.dataset.kaldi_io as kaldi_io

torchaudio.set_audio_backend("sox")


def parse_opts():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--num_mel_bins',
                        default=80,
                        type=int,
                        help='Number of triangular mel-frequency bins')
    parser.add_argument('--frame_length',
                        type=int,
                        default=25,
                        help='Frame length in milliseconds')
    parser.add_argument('--frame_shift',
                        type=int,
                        default=10,
                        help='Frame shift in milliseconds')
    parser.add_argument('--dither',
                        type=int,
                        default=0.0,
                        help='Dithering constant (0.0 means no dither)')
    parser.add_argument('--segments',
                        type=int,
                        default=None,
                        help='segments file')
    parser.add_argument('wav_scp', help='wav scp file')
    parser.add_argument('out_ark', help='output ark file')
    parser.add_argument('out_scp', help='output scp file')
    args = parser.parse_args()
    return args


def load_wav_scp(wav_scp_file):
    wav_list = []
    with open(wav_scp_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_list.append((arr[0], arr[1]))
    return wav_list


if __name__ == '__main__':
    args = parse_opts()
    wav_list = load_wav_scp(args.wav_scp)
    with open(args.out_ark, 'wb') as ark_fout, \
         open(args.out_scp, 'w', encoding='utf8') as scp_fout:
        for item in wav_list:
            key, wav_path = item
            waveform, sample_rate = torchaudio.load_wav(wav_path)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=args.num_mel_bins,
                              frame_length=args.frame_length,
                              frame_shift=args.frame_shift,
                              dither=args.dither,
                              energy_floor=0.0,
                              sample_frequency=sample_rate)
            mat = mat.detach().numpy()
            kaldi_io.write_ark_scp(key, mat, ark_fout, scp_fout)
