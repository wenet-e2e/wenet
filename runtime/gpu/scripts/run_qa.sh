#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

export CUDA_VISIBLE_DEVICES="0"
stage=0
stop_stage=2

# aishell small offline model
pretrained_model_link=https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20211025_conformer_exp.tar.gz
# wenetspeech large offline model
# pretrained_model_link=https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20211025_conformer_exp.tar.gz
# aishell2 small streaming u2pp model
# pretrained_model_link=http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz
pretrained_model_name=20211025_conformer_exp
model_repo_path=$(pwd)/../model_repo

model_dir=$(pwd)/${pretrained_model_name}
onnx_model_dir=$(pwd)/${pretrained_model_name}_onnx
mkdir -p $onnx_model_dir

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   echo "export to onnx files"
   wget ${pretrained_model_link} --no-check-certificate
   tar zxvf ${pretrained_model_name}.tar.gz

   cd ../../../
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 wenet/bin/export_onnx_gpu.py \
           --config=$model_dir/train.yaml \
           --checkpoint=$model_dir/final.pt \
           --cmvn_file=$model_dir/global_cmvn \
           --ctc_weight=0.5 \
           --output_onnx_dir=$onnx_model_dir \
           --fp16 || exit 1
   cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir
   cd -
fi

# For streaming model
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
   echo "export to onnx files"
   wget ${pretrained_model_link} --no-check-certificate
   tar zxvf ${pretrained_model_name}.tar.gz

   cd ../../../
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 wenet/bin/export_onnx_gpu.py \
           --config=$model_dir/train.yaml \
           --checkpoint=$model_dir/final.pt \
           --cmvn_file=$model_dir/global_cmvn \
           --ctc_weight=0.5 \
           --output_onnx_dir=$onnx_model_dir \
           --streaming \
           --fp16 || exit 1
   cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir
   cd -
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   echo "prepare model repo for triton"

   python3 ./convert.py --config=$onnx_model_dir/train.yaml --vocab=$onnx_model_dir/words.txt \
        --model_repo=$model_repo_path --onnx_model_dir=$onnx_model_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   echo "launch triton server"
   tritonserver --model-repository $model_repo_path \
                --pinned-memory-pool-byte-size=512000000 \
                --cuda-memory-pool-byte-size=0:1024000000
fi
