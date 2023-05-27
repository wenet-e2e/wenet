#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-04-27] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import random
import tarfile

random.seed(1024)

# parse arg from command line
datalist = os.sys.argv[1]
datatype = os.sys.argv[2]
num_gpus = int(os.sys.argv[3])
num_samples_per_tar = int(os.sys.argv[4])  # only used in shard mode
new_datalist = os.sys.argv[5]

assert datatype in ["shard", "raw"]


filtered_list = []
with open(datalist, "r") as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    if datatype == "raw":
        valid_num = len(lines) // num_gpus * num_gpus
        random.shuffle(lines)
        filtered_list = lines[:valid_num]
    else:
        for line in lines:
            cnt = 0
            with open(line, "rb") as tar:
                stream = tarfile.open(fileobj=tar, mode="r|*")
                for tarinfo in stream:
                    name = tarinfo.name
                    pos = name.rfind('.')
                    assert pos > 0
                    prefix, postfix = name[:pos], name[pos + 1:]
                    if postfix == 'txt':
                        cnt += 1
            if cnt == num_samples_per_tar:
                filtered_list.append(line)
        valid_num = len(filtered_list) // num_gpus * num_gpus
        random.shuffle(filtered_list)
        filtered_list = filtered_list[:valid_num]
    filtered_list.sort()
    print("before filter: {} after filter: {}".format(len(lines), len(filtered_list)))

with open(new_datalist, "w") as f:
    for line in filtered_list:
        f.writelines("{}\n".format(line))
