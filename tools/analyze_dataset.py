#!/usr/bin/env python3

# Copyright (c) 2022 Tsinghua Univ. (authors: Xingchen Song)
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
"""
Analyze Dataset, Duration/TextLength/Speed etc.

Usage:
. ./path.sh && python3 tools/analyze_dataset.py \
    --data_type "shard" \
    --data_list data/test/data.list \
    --output_dir exp/analyze_test \
    --num_thread 32
"""

import os
import json
import math
import time
import numpy
import logging
import librosa
import tarfile
import argparse
import torchaudio
import multiprocessing

from wenet.utils.file_utils import read_lists
from wenet.dataset.processor import AUDIO_FORMAT_SETS


def get_args():
    parser = argparse.ArgumentParser(description='Analyze dataset')
    parser.add_argument('--data_type',
                        default='wav_scp',
                        choices=['wav_scp', 'raw', 'shard'],
                        help='dataset type')
    parser.add_argument('--output_dir',
                        type=str,
                        default="exp",
                        help='write info to output dir')
    parser.add_argument('--data_list',
                        default=None,
                        help='used in raw/shard mode')
    parser.add_argument('--wav_scp', default=None, help='used in wav_scp mode')
    parser.add_argument('--text', default=None, help='used in wav_scp mode')
    parser.add_argument('--num_thread',
                        type=int,
                        default=4,
                        help='number of threads')
    args = parser.parse_args()
    print(args)
    return args


def analyze(datas, output_file, thread_id):
    with open(output_file, "w", encoding='utf8') as f:
        for i, data in enumerate(datas):
            if type(data['wav']) is numpy.ndarray:
                y, sample_rate = data['wav'], data['sample_rate']
                data['wav'] = "None"  # NOTE(xcsong): Do not save wav.
            elif type(data['wav'] is str):
                y, sample_rate = librosa.load(data['wav'], sr=16000)
            data['dur'] = len(y) / sample_rate
            data['txt_length'] = len(data['txt'])
            data['speed'] = data['txt_length'] / data['dur']
            # Trim the beginning and ending silence
            _, index = librosa.effects.trim(y, top_db=30)
            data['leading_sil'] = librosa.get_duration(
                y=y[:index[0]], sr=16000) * 1000 if index[0] > 0 else 0
            data['trailing_sil'] = librosa.get_duration(
                y=y[index[1]:], sr=16000) * 1000 if index[1] < len(y) else 0
            data_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(data_str))
            if thread_id == 0 and i % 100 == 0:
                logging.info("\tThread-{}: processed {}/{}".format(
                    thread_id, i, len(datas)))


def read_tar(file):
    try:
        with tarfile.open(fileobj=open(file, "rb"), mode="r|*") as stream:
            prev_prefix = None
            data = {}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    data['key'] = prev_prefix
                    if valid:
                        yield data
                    data = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            data['txt'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix in AUDIO_FORMAT_SETS:
                            waveform, sample_rate = torchaudio.load(file_obj)
                            # single channel
                            data['wav'] = waveform.numpy()[0, :]
                            data['sample_rate'] = sample_rate
                        else:
                            data[postfix] = file_obj.read()
                    except Exception as ex:
                        valid = False
                        logging.warning('error: {} when parse {}'.format(
                            ex, name))
                prev_prefix = prefix
            # The last data in tar
            if prev_prefix is not None:
                data['key'] = prev_prefix
                yield data
    except Exception as ex:
        logging.warning('tar_file error: {} when processing {}'.format(
            ex, file))


def main():
    start_time = time.time()
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/partition", exist_ok=True)
    datas = [[] for i in range(args.num_thread)]

    logging.info("Stage-1: Loading data.list OR wav.scp...")
    if args.data_type == "shard":
        assert args.data_list is not None
        lists = read_lists(args.data_list)
        # partition
        total = 0
        for line in lists:
            for data in read_tar(line):
                datas[total % args.num_thread].append(data)
                total = total + 1
    elif args.data_type == "raw":
        assert args.data_list is not None
        lists = read_lists(args.data_list)
        # partition
        for i, line in enumerate(lists):
            data = json.loads(line)
            datas[i % args.num_thread].append(data)
    elif args.data_type == "wav_scp":
        assert args.wav_scp is not None
        assert args.text is not None
        wavs, texts = {}, {}
        # wavs
        for line in read_lists(args.wav_scp):
            line = line.strip().split()
            wavs[line[0]] = line[1]
        # texts
        for line in read_lists(args.text):
            line = line.strip().split(maxsplit=1)
            texts[line[0]] = line[1]
        sorted(wavs)
        sorted(texts)
        # partition
        for i, (key1, key2) in enumerate(zip(wavs, texts)):
            assert key1 == key2
            datas[i % args.num_thread].append({
                'key': key1,
                "wav": wavs[key1],
                "txt": texts[key1]
            })

    logging.info("Stage-2: Start Analyze")
    # threads
    pool = multiprocessing.Pool(processes=args.num_thread)
    for i in range(args.num_thread):
        output_file = os.path.join(args.output_dir, "partition",
                                   "part-{}".format(i))
        pool.apply_async(analyze, (datas[i], output_file, i))
    pool.close()
    pool.join()

    logging.info("Stage-3: Sort and Write Result")
    datas = []
    for i in range(args.num_thread):
        output_file = os.path.join(args.output_dir, "partition",
                                   "part-{}".format(i))
        with open(output_file, "r", encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                datas.append(data)
    total_dur = sum([x['dur'] for x in datas])
    total_len = sum([x['txt_length'] for x in datas])
    total_leading_sil = sum([x['leading_sil'] for x in datas])
    total_trailing_sil = sum([x['trailing_sil'] for x in datas])
    num_datas = len(datas)
    names = [
        'key', 'dur', 'txt_length', 'speed', 'leading_sil', 'trailing_sil'
    ]
    units = ['', 's', '', 'char/s', 'ms', 'ms']
    avgs = [
        0, total_dur / num_datas, total_len / num_datas, total_len / total_dur,
        total_leading_sil / num_datas, total_trailing_sil / num_datas
    ]
    stds = [
        0,
        sum([(x['dur'] - avgs[1])**2 for x in datas]),
        sum([(x['txt_length'] - avgs[2])**2 for x in datas]),
        sum([(x['txt_length'] / x['dur'] - avgs[3])**2 for x in datas]),
        sum([(x['leading_sil'] - avgs[4])**2 for x in datas]),
        sum([(x['trailing_sil'] - avgs[5])**2 for x in datas])
    ]
    stds = [math.sqrt(x / num_datas) for x in stds]
    parts = ['max', 'P99', 'P75', 'P50', 'P25', 'min']
    index = [
        num_datas - 1,
        int(num_datas * 0.99),
        int(num_datas * 0.75),
        int(num_datas * 0.50),
        int(num_datas * 0.25), 0
    ]

    with open(args.output_dir + "/analyze_result_brief", "w",
              encoding='utf8') as f:
        for i, (name, unit, avg,
                std) in enumerate(zip(names, units, avgs, stds)):
            if name == 'key':
                continue
            f.write("==================\n")

            datas.sort(key=lambda x: x[name])
            for p, j in zip(parts, index):
                f.write("{} {}: {:.3f} {} (wav_id: {})\n".format(
                    p, name, datas[j][name], unit, datas[j]['key']))
            f.write("avg {}: {:.3f} {}\n".format(name, avg, unit))
            f.write("std {}: {:.3f}\n".format(name, std))
    os.system("cat {}".format(args.output_dir + "/analyze_result_brief"))

    datas.sort(key=lambda x: x['dur'])
    with open(args.output_dir + "/analyze_result", "w", encoding='utf8') as f:
        for data in datas:
            f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

    end_time = time.time()
    logging.info("Time Cost: {:.3f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()
