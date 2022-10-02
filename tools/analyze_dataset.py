#!/usr/bin/env python3

# Copyright (c) 2022 Horizon Inc. (authors: Xingchen Song)
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

"""Analyze Dataset, Duration/TextLength/Speed etc."""

import argparse
import logging
import queue
import threading
import librosa

from wenet.utils.file_utils import read_lists


def get_args():
    parser = argparse.ArgumentParser(description='Analyze dataset')
    parser.add_argument('--data_type',
                        default='wav_scp',
                        choices=['wav_scp', 'raw', 'shard'],
                        help='dataset type')
    parser.add_argument('--data_list', default=None,
                        help='used in raw/shard mode')
    parser.add_argument('--wav_scp', default=None,
                        help='used in wav_scp mode')
    parser.add_argument('--text', default=None,
                        help='used in wav_scp mode')
    parser.add_argument('--num_thread', type=int,
                        default=4, help='number of threads')
    args = parser.parse_args()
    print(args)
    return args


def query_dict(wavs_queue, datas, wavs, texts):
    while not wavs_queue.empty():
        key = wavs_queue.get()
        if key in texts.keys():
            y, sample_rate = librosa.load(wavs[key], sr=16000)
            dur = len(y) / sample_rate
            text_length = len(texts[key])
            speed = text_length / dur
            # Trim the beginning and ending silence
            _, index = librosa.effects.trim(y, top_db=30)
            leading_sil = librosa.get_duration(
                y=y[:index[0]], sr=16000) * 1000 if index[0] > 0 else 0
            trailing_sil = librosa.get_duration(
                y=y[index[1]:], sr=16000) * 1000 if index[1] < len(y) else 0
            datas.append([key, dur, text_length, speed,
                          leading_sil, trailing_sil])
        else:
            logging.warning("{} not in text, pass".format(key))


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # List of [id, duration, textlenghth, speed, leading_sil, trailing_sil]
    datas = []
    threads = []
    if args.data_type == "shard":
        assert args.data_list is not None
        lists = read_lists(args.data_list)
        raise NotImplementedError("Feel free to make a PR :)")
    elif args.data_type == "raw":
        assert args.data_list is not None
        lists = read_lists(args.data_list)
        raise NotImplementedError("Feel free to make a PR :)")
    elif args.data_type == "wav_scp":
        assert args.wav_scp is not None
        assert args.text is not None
        logging.info("Start Analyze {}".format(args.wav_scp))
        wavs, texts = {}, {}
        wavs_queue = queue.Queue()
        # wavs & wavs_queue
        for line in read_lists(args.wav_scp):
            line = line.strip().split()
            wavs[line[0]] = line[1]
            wavs_queue.put(line[0])
        # texts
        for line in read_lists(args.text):
            line = line.strip().split(maxsplit=1)
            texts[line[0]] = line[1]
        # threads
        for i in range(args.num_thread):
            t = threading.Thread(target=query_dict,
                                 args=(wavs_queue, datas, wavs, texts))
            threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_dur = sum([x[1] for x in datas])
    total_len = sum([x[2] for x in datas])
    total_leading_sil = sum([x[4] for x in datas])
    total_trailing_sil = sum([x[5] for x in datas])
    num_datas = len(datas)
    names = ['key', 'duration', 'text_length', 'speed',
             'leading_sil', 'trailing_sil']
    units = ['', 's', '', 'char/s', 'ms', 'ms']
    avgs = [0, total_dur / num_datas, total_len / num_datas,
            total_len / total_dur, total_leading_sil / num_datas,
            total_trailing_sil / num_datas]
    parts = ['max', 'P99', 'P75', 'P50', 'P25', 'min']
    index = [num_datas - 1, int(num_datas * 0.99), int(num_datas * 0.75),
             int(num_datas * 0.50), int(num_datas * 0.25), 0]

    for i, (name, unit, avg) in enumerate(zip(names, units, avgs)):
        if name == 'key':
            continue
        logging.info("==================")

        def f(i=i):
            """ Avoid late binding, see:
                https://stackoverflow.com/questions/3431676/creating-functions-or-lambdas-in-a-loop-or-comprehension
            """
            return i

        datas.sort(key=lambda x: x[f()])
        for p, j in zip(parts, index):
            logging.info("{} {}: {:.3f} {} (wav_id: {})".format(
                p, name, datas[j][f()], unit, datas[j][0]))
        logging.info("avg {}: {:.3f} {}".format(
            name, avg, unit))


if __name__ == '__main__':
    main()
