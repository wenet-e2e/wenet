# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


def get_args():
    parser = argparse.ArgumentParser(description='sum up prediction wer')
    parser.add_argument('--job_num',
                        type=int,
                        default=8,
                        help='number of total split dir')
    parser.add_argument('--dir_split',
                        required=True,
                        help='the path to the data_list dir '
                        'eg data/train/wenet1k_good_split_60/')
    parser.add_argument('--label',
                        type=int,
                        default=0,
                        help='if ture, label file will also be considered.')
    parser.add_argument('--hypo_name',
                        type=str,
                        required=True,
                        help='the hypothesis path.  eg. /hypothesis_0.txt ')
    parser.add_argument(
        '--wav_dir',
        type=str,
        required=True,
        help='the wav dir path.  eg. data/train/wenet_1k_untar/ ')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_list_dir = args.dir_split
    num_lists = args.job_num
    hypo = args.hypo_name
    # wav_dir is the directory where your pair of ID.scp
    # (the audio file ) and ID.txt (the optional label file ) file stored.
    # We assumed that you have generated this dir in data processing steps.
    wav_dir = args.wav_dir
    label = args.label

    print("data_list_path is", data_list_dir)
    print("num_lists is", num_lists)
    print("hypo is", hypo)
    print("wav_dir is", wav_dir)

    i = num_lists
    c = 0
    hypo_path = data_list_dir + "data_sublist" + str(i) + hypo
    output_wav = data_list_dir + "data_sublist" + str(i) + "/wav.scp"
    output_label = data_list_dir + "data_sublist" + str(i) + "/label.txt"
    # bad lines are just for debugging
    output_bad_lines = data_list_dir + "data_sublist" + str(
        i) + "/bad_line.txt"

    with open(hypo_path, 'r', encoding="utf-8") as reader:
        hypo_lines = reader.readlines()

    wavs = []
    labels = []
    bad_files = []
    for x in hypo_lines:
        c += 1
        file_id = x.split()[0]

        label_path = wav_dir + file_id + ".txt"
        wav_path = wav_dir + file_id + ".wav\n"
        wav_line = file_id + " " + wav_path
        wavs.append(wav_line)
        if label:
            try:
                with open(label_path, 'r', encoding="utf-8") as reader1:
                    label_line = reader1.readline()
            except OSError as e:
                bad_files.append(label_path)

            label_line = file_id + " " + label_line + "\n"
            labels.append(label_line)

    with open(output_wav, 'w', encoding="utf-8") as writer2:
        for wav in wavs:
            writer2.write(wav)
    with open(output_bad_lines, 'w', encoding="utf-8") as writer4:
        for line in bad_files:
            writer4.write(line)
    if label:
        with open(output_label, 'w', encoding="utf-8") as writer3:
            for label in labels:
                writer3.write(label)


if __name__ == '__main__':
    main()
