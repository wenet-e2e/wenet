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
    parser.add_argument('-t',
                        '--show_tokens_info',
                        action='store_true',
                        help='whether to output token(word) level information'
                        ', such times/confidence')
    parser.add_argument('--align',
                        action='store_true',
                        help='force align the input audio and transcript')
    parser.add_argument('--label', type=str, help='the input label to align')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model(args.language, args.model_dir)
    if args.align:
        result = model.align(args.audio_file, args.label)
    else:
        result = model.transcribe(args.audio_file, args.show_tokens_info)
    print(result)


if __name__ == "__main__":
    main()
