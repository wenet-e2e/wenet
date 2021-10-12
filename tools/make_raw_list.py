#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import sys
import json

wav_file = sys.argv[1]
text_file = sys.argv[2]
output_file = sys.argv[3]

wav_table = {}
with open(wav_file, 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.strip().split()
        assert len(arr) == 2
        wav_table[arr[0]] = arr[1]


with open(text_file, 'r', encoding='utf8') as fin, \
     open(output_file, 'w', encoding='utf8') as fout:
    for line in fin:
        arr = line.strip().split(maxsplit=1)
        key = arr[0]
        txt = arr[1] if len(arr) > 1 else ''
        assert key in wav_table
        wav = wav_table[key]
        line = dict(key=key, wav=wav, txt=txt)
        json_line = json.dumps(line, ensure_ascii=False)
        fout.write(json_line + '\n')
