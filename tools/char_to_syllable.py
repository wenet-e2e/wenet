# Copyright (c) 2020 Mobvoi Inc. (authors: Di Wu)
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
import logging

from pypinyin import pinyin, lazy_pinyin, Style

def parse_opts():
    parser = argparse.ArgumentParser(description='convert char to syllable')
    parser.add_argument('--char_text',
                        required=True,
                        type=str,
                        help='source char text')
    parser.add_argument('--syllable_text',
                        required=True,
                        type=str,
                        help='dst syllable text')
    parser.add_argument('--tone',
                        action='store_true',
                        default=False,
                        help='use tone syllable')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_opts()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    with open(args.char_text, 'r') as src, open(args.syllable_text, 'w') as dst:
        for line in src:
            key = line.split(' ')[0]
            line = line.split(' ')[1]
            line = line.strip()
            if args.tone:
                syllable = pypinyin(line)
            else:
                syllable = lazy_pinyin(line)
            dst.writelines('{} {}\n'.format(key, ' '.join(syllable)))
