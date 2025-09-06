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

from wenet.cli.model import load_model
from wenet.cli.punc_model import load_model as load_punc_model  # noqa


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('audio_file', help='audio file to transcribe')
    parser.add_argument('-m',
                        '--model',
                        default='wenetspeech',
                        help='model name or local model dir, built in models:'
                        '[wenetspeech|paraformer|firered|whisper*]')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('-t',
                        '--show_tokens_info',
                        action='store_true',
                        help='whether to output token(word) level information'
                        ', such times/confidence')
    parser.add_argument('--align',
                        action='store_true',
                        help='force align the input audio and transcript')
    parser.add_argument('--label', type=str, help='the input label to align')
    parser.add_argument('--beam', type=int, default=5, help="beam size")
    parser.add_argument('--context_path',
                        type=str,
                        default=None,
                        help='context list file')
    parser.add_argument('--context_score',
                        type=float,
                        default=6.0,
                        help='context score')
    parser.add_argument('--punc', action='store_true', help='context score')

    parser.add_argument('-pm',
                        '--punc_model_dir',
                        default=None,
                        help='specify your own punc model dir')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # TODO(Binbin Zhang): Add other feature, such as device, paraformer, ...
    model = load_model(args.model, device=args.device)
    result = model.transcribe(args.audio_file)
    print(result.text)


if __name__ == "__main__":
    main()
