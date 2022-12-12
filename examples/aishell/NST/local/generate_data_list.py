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
import os
import random

def get_args():
    parser = argparse.ArgumentParser(description='generate data.list file ')
    parser.add_argument('--tar_dir', help='path for tar file')
    parser.add_argument('--supervised_data_list',
                        help='path for supervised data list')
    parser.add_argument('--pseudo_data_ratio',
                        help='ratio of pseudo data, '
                             '0 means none pseudo data, '
                             '1 means all using pseudo data.')
    parser.add_argument('--out_data_list', help='output path for data list')
    args = parser.parse_args()
    return args


#python generate_data_list.py --tar_dir wenet_1k_tar_10_11_nss7 --supervised_data_list data/train/data_aishell.list --pseudo_data_ratio 0.75 --out_data_list test_g1.list

def main():
    args = get_args()
    target_dir = args.tar_dir
    # target_dir = "data/train/wenet_bad_tar_10_7_nst3_r6"
    pseudo_data_list = os.listdir(target_dir)
    output_file = args.out_data_list
    # output_file = "data/train/wenet_bad_tar_10_7_nst3_r6.list"
    pseudo_data_ratio = args.pseudo_data_ratio
    supervised_path = args.supervised_data_list
    with open(supervised_path,"r") as reader:
        supervised_data_list = reader.readlines()
    pseudo_len = len(pseudo_data_list)
    supervised_len = len(supervised_data_list)
    random.shuffle(pseudo_data_list)
    random.shuffle(supervised_data_list)

    cur_ratio = pseudo_len / (pseudo_len + supervised_len)
    if cur_ratio < pseudo_data_ratio:
        pseudo_to_super_datio = pseudo_data_ratio / (1 - pseudo_data_ratio)
        supervised_len = int(pseudo_len / pseudo_to_super_datio)
    elif cur_ratio > pseudo_data_ratio:
        super_to_pseudo_datio = (1 - pseudo_data_ratio) / pseudo_data_ratio
        pseudo_len = int(supervised_len / super_to_pseudo_datio)

    fused_list = pseudo_data_list[:pseudo_len] + supervised_data_list[:supervised_len]

    with open(output_file, "w") as writer:
        for line in fused_list:
            writer.write(target_dir + "/" + line + "\n")


if __name__ == '__main__':
    main()
