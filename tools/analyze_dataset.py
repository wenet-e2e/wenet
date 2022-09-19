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

import torchaudio

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
            waveform, sample_rate = torchaudio.load(wavs[key])
            dur = len(waveform[0]) / sample_rate
            text_length = len(texts[key])
            speed = text_length / dur
            datas.append([dur, text_length, speed, key])
        else:
            logging.warning("{} not in text, pass".format(key))


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    datas = []  # List of [duration, textlenghth, speed, id]
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

    total_dur = sum(map(lambda x: x[0], datas))
    total_len = sum(map(lambda x: x[1], datas))
    num_datas = len(datas)
    logging.info("==================")
    datas.sort(key=lambda x: x[0])  # sort by duration
    logging.info("max duration: {:.3f} s (wav_id: {})".format(
        datas[-1][0], datas[-1][3]))
    logging.info("P99 duration: {:.3f} s".format(
        datas[int(num_datas * 0.99)][0]))
    logging.info("P75 duration: {:.3f} s".format(
        datas[int(num_datas * 0.75)][0]))
    logging.info("P50 duration: {:.3f} s".format(
        datas[int(num_datas * 0.5)][0]))
    logging.info("P25 duration: {:.3f} s".format(
        datas[int(num_datas * 0.25)][0]))
    logging.info("min duration: {:.3f} s (wav_id: {})".format(
        datas[0][0], datas[0][-1]))
    logging.info("avg duration: {:.3f} s".format(
        total_dur / len(datas)))
    logging.info("==================")
    datas.sort(key=lambda x: x[1])  # sort by text length
    logging.info("max text length: {} (wav_id: {})".format(
        datas[-1][1], datas[-1][3]))
    logging.info("P99 text length: {}".format(
        datas[int(num_datas * 0.99)][1]))
    logging.info("P75 text length: {}".format(
        datas[int(num_datas * 0.75)][1]))
    logging.info("P50 text length: {}".format(
        datas[int(num_datas * 0.5)][1]))
    logging.info("P25 text length: {}".format(
        datas[int(num_datas * 0.25)][1]))
    logging.info("min text length: {} (wav_id: {})".format(
        datas[0][1], datas[0][-1]))
    logging.info("avg text length: {:.3f}".format(
        total_len / len(datas)))
    logging.info("==================")
    datas.sort(key=lambda x: x[2])  # sort by speed
    logging.info("max speed: {:.3f} char/s (wav_id: {})".format(
        datas[-1][2], datas[-1][3]))
    logging.info("P99 speed: {:.3f} char/s".format(
        datas[int(num_datas * 0.99)][2]))
    logging.info("P75 speed: {:.3f} char/s".format(
        datas[int(num_datas * 0.75)][2]))
    logging.info("P50 speed: {:.3f} char/s".format(
        datas[int(num_datas * 0.5)][2]))
    logging.info("P25 speed: {:.3f} char/s".format(
        datas[int(num_datas * 0.25)][2]))
    logging.info("min speed: {:.3f} char/s (wav_id: {})".format(
        datas[0][2], datas[0][-1]))
    logging.info("avg speed: {:.3f} char/s".format(
        total_len / total_dur))


if __name__ == '__main__':
    main()
