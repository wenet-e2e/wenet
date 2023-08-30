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

from wenetruntime.decoder import Decoder
from _wenet import wenet_set_log_level as set_log_level  # noqa


def get_args():
    parser = argparse.ArgumentParser(description='wenet')
    parser.add_argument('--language',
                        default='chs',
                        choices=['chs', 'en'],
                        help='select language')
    parser.add_argument('-c',
                        '--chunk_size',
                        default=-1,
                        type=int,
                        help='set decoding chunk size')
    parser.add_argument('-v',
                        '--verbose',
                        default=0,
                        type=int,
                        help='set log(glog backend) level')
    parser.add_argument('audio', help='input audio file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    set_log_level(args.verbose)
    decoder = Decoder(lang=args.language)
    decoder.set_chunk_size(args.chunk_size)
    result = decoder.decode(args.audio)
    print(result)


if __name__ == '__main__':
    main()
