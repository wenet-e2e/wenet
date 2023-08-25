# Copyright (c) 2023  Binbin Zhang(binbzha@qq.com)
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

from .decoder import Decoder


def get_args():
    parser = argparse.ArgumentParser(description='wenet')
    parser.add_argument('--language',
                        default='chs',
                        choices=['chs', 'en'],
                        help='select language')
    parser.add_argument('audio', help='input audio file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    decoder = Decoder(lang=args.language)
    result = decoder.decode(args.audio)
    print(result)


if __name__ == '__main__':
    main()
