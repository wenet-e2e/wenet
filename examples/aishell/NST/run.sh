#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

iter_num=2
stage=1
stop_stage=1
pseudo_data_ratio=0.75
dir=exp/conformer_test_fully_supervised
data_list=data_aishell.list
supervised_data_list=data_aishell.list
unsupervised_data_list=wenet_1khr.list
dir_split=wenet_split_60_test/
out_data_list=data/train/wenet_1khr_nst0.list
num_split=1
. tools/parse_options.sh || exit 1;

# Stage 1 trains the initial teacher and generates initial pseudo-labels.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "******** stage 1 training the intial teacher ********"
  bash run_nst.sh --dir $dir \
  --data_list $data_list \
  --supervised_data_list $supervised_data_list \
  --unsupervised_data_list $unsupervised_data_list \
  --dir_split $dir_split\
  --out_data_list $out_data_list \
  --enable_nst 0 \
  --pseudo_data_ratio pseudo_data_ratio \
  --num_split $num_split

fi

# Stage 2 trains the nst iterations.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  for ((i = 0; i < $iter_num; ++i)); do
  {
    echo "******** stage 2 training nst iteration number $i ********"
    bash run_nst.sh --dir exp/conformer_nst${i+1} \
      --supervised_data_list data_aishell.list \
      --data_list wenet_1khr_nst${i}.list \
      --enable_nst 1 \
      --job_num 0 \
      --num_split $num_split \
      --hypo_name hypothesis_nst${i+1}.txt \
      --untar_dir wenet_1khr_untar_nst${i+1}/ \
      --tar_dir wenet_1khr_tar_nst${i+1}/ \
      --out_data_list wenet_1khr_nst${i+1}.list \
      --pseudo_data_ratio $pseudo_data_ratio

  }
  done

fi
