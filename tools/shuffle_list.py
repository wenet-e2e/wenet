#!/usr/bin/env python3
# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import sys

parser = argparse.ArgumentParser(description='shuffle input file by line')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--input', help='input file')
parser.add_argument('--output', help='output file')
args = parser.parse_args()

random.seed(args.seed)

if args.input is not None:
    fin = open(args.input, 'r', encoding='utf8')
else:
    fin = sys.stdin

lines = fin.readlines()
random.shuffle(lines)

if args.output is not None:
    fout = open(args.output, 'w', encoding='utf8')
else:
    fout = sys.stdout

try:
    fout.writelines(lines)
except Exception:
    pass
if args.input is not None:
    fin.close()
if args.output is not None:
    fout.close()
