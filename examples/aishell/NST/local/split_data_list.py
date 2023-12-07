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

import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_nums',
                        type=int,
                        default=8,
                        help='number of total split jobs')
    parser.add_argument('--data_list_path',
                        required=True,
                        help='the path to the data.list file')
    parser.add_argument('--output_dir',
                        required=True,
                        help='path to output dir, '
                        'eg --output_dir=data/train/aishell_split_60')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_list_path = args.data_list_path
    num_lists = args.job_nums
    output_dir = args.output_dir

    print("data_list_path is", data_list_path)
    print("num_lists is", num_lists)
    print("output_dir is", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(data_list_path, 'r', encoding="utf-8") as reader:
        data_list_we = reader.readlines()

    # divide data.list equally
    len_d = int(len(data_list_we) / num_lists)
    rest_lines = data_list_we[num_lists * len_d:]
    rest_len = len(rest_lines)
    print("total num of lines", len(data_list_we), "rest len is", rest_len)

    # generate N sublist
    for i in range(num_lists):
        print("current dir num", i)
        out_put_sub_dir = output_dir + "/" + "data_sublist" + str(i) + "/"
        os.makedirs(out_put_sub_dir, exist_ok=True)
        output_list = out_put_sub_dir + "data_list"

        with open(output_list, 'w', encoding="utf-8") as writer:

            new_list = data_list_we[i * len_d:(i + 1) * len_d]
            if i < rest_len:
                new_list.append(rest_lines[i])
            for x in new_list:
                # output list
                writer.write(x)


if __name__ == '__main__':
    main()
