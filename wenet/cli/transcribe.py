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

from wenet.cli.model import Model
from wenet.cli.paraformer_model import Paraformer


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('audio_file', help='audio file to transcribe')
    parser.add_argument('--language',
                        choices=[
                            'chinese',
                            'english',
                            'chinese-paraformer',
                        ],
                        default='chinese',
                        help='language type')
    parser.add_argument('--model_dir', default='', help='wenet jit model dirs')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.language == 'chinese-paraformer':
        model = Paraformer(args.model_dir)
    else:
        model = Model(args.language)
    result = model.transcribe(args.audio_file)
    print(result)


if __name__ == "__main__":
    main()
