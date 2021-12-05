#!/usr/bin/env python3
# Copyright (c) 2021 Mobvoi Inc. (Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate segmented wav.scp')
    parser.add_argument('--segments', required=True, help='segments file')
    parser.add_argument('--input',
                        required=True,
                        help='origin wav.scp that not segmented')
    parser.add_argument('--output',
                        required=True,
                        help='output segmented wav.scp')
    wav_dic = {}
    args = parser.parse_args()
    ori_wav = args.input
    segment_file = args.segments
    wav_scp = args.output
    with open(ori_wav, 'r') as ori:
        for l in ori:
            item = l.strip().split()
            wav_dic[item[0]] = item[1]
    with open(wav_scp, 'w') as f, open(segment_file, 'r') as sgement:
        for l in sgement:
            item = l.strip().split()
            if item[1] in wav_dic:
                item[1] = wav_dic[item[1]]
                f.write("{} {},{},{}\n".format(item[0], item[1], item[2], item[3]))
