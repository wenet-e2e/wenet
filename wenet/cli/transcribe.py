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
from wenet.cli.paraformer_model import load_model as load_paraformer
from wenet.cli.punc_model import load_model as load_punc_model


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('audio_file', help='audio file to transcribe')
    parser.add_argument('-l',
                        '--language',
                        choices=[
                            'chinese',
                            'english',
                        ],
                        default='chinese',
                        help='language type')
    parser.add_argument('-m',
                        '--model_dir',
                        default=None,
                        help='specify your own model dir')
    parser.add_argument('-g',
                        '--gpu',
                        type=int,
                        default='-1',
                        help='gpu id to decode, default is cpu.')
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
    parser.add_argument('--paraformer',
                        action='store_true',
                        help='whether to use the best chinese model')
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

    if args.paraformer:
        model = load_paraformer(args.model_dir, args.gpu, args.device)
    else:
        model = load_model(args.language, args.model_dir, args.gpu, args.beam,
                           args.context_path, args.context_score, args.device)
    punc_model = None
    if args.punc:
        punc_model = load_punc_model(args.punc_model_dir, args.gpu,
                                     args.device)
    if args.align:
        result = model.align(args.audio_file, args.label)
    else:
        result = model.transcribe(args.audio_file, args.show_tokens_info)
        if args.punc:
            assert punc_model is not None
            result['text_with_punc'] = punc_model(result['text'])
    print(result)


if __name__ == "__main__":
    main()
