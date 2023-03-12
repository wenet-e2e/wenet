#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0

current_dir=$(pwd)
stage=0
stop_stage=0
. ./path.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  cd $current_dir/data/
  [ ! -z vkw_v1.1.zip ] && echo "wget vkw challenge data to this directory" && exit 0
  [ ! -z vkw ] && unzip vkw_v1.1.zip
  cd $current_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  x=train
  [ ! -f data/${x}/text ] && echo "vkw trainset is missing, wget to this directory" && exit 0
fi

echo "$0: vkw  data preparation succeeded"
