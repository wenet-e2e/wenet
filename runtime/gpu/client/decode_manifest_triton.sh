#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#

exp_dir=model_repo_a10_aliyun

mkdir -p $exp_dir
for num_task in 60 40
do
  python3 decode_manifest_triton.py \
    --server-addr localhost \
    --num-tasks $num_task \
    --log-interval 20 \
    --model-name attention_rescoring \
    --manifest-filename /myworkspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz \
    --compute-cer 
  mv rtf.txt $exp_dir/rtf-${num_task}.txt
  mv errs-aishell_cuts_test.txt $exp_dir/errs-aishell_cuts_test-${num_task}.txt
done
