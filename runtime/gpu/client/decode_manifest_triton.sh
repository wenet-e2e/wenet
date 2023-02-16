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

serveraddr=localhost
manifest_path=/myworkspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz
exp_dir=model_repo_stateful_trt_exp1
mkdir -p $exp_dir

for num_task in 20 40 60 80
do
python3 decode_manifest_triton.py \
    --server-addr $serveraddr \
    --streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks $num_task \
    --manifest-filename $manifest_path
mv rtf.txt $exp_dir/rtf-${num_task}-streaming.txt
mv errs-aishell_cuts_test.txt $exp_dir/errs-aishell_cuts_test-${num_task}.txt
done

for num_task in 500
do
# For simulate streaming mode wenet server
python3 decode_manifest_triton.py \
    --server-addr $serveraddr \
    --simulate-streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks $num_task \
    --manifest-filename $manifest_path
mv rtf.txt $exp_dir/rtf-${num_task}-simulate-streaming.txt
mv errs-aishell_cuts_test.txt $exp_dir/errs-aishell_cuts_test-${num_task}.txt
done

python3 stats_summary.py
mv stats.json $exp_dir/
mv stats_summary.txt $exp_dir/

perf_analyzer -m streaming_wenet -b 1 -a -p 10000 --concurrency-range 50:201:50 -i gRPC --input-data=./online_input.json  -u $serveraddr:8001 -f $exp_dir/log.txt --streaming
perf_analyzer -m encoder -b 1 -a -p 5000 --concurrency-range 100:500:100 -i gRPC -u $serveraddr:8001 --streaming -f $exp_dir/log_encoder.txt
