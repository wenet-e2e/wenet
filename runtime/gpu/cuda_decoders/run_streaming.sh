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
stage=-1
stop_stage=3

#<wenet_onnx_gpu_models>
onnx_model_dir=$(pwd)/aishell_u2pp_onnx
mkdir -p $onnx_model_dir
# modify model parameters according to your own model
D_MODEL=256
VOCAB_SIZE=4233
CACHE_SIZE=80 # decoding_chunk_size * num_decoding_left_chunks, 16 * 5
NUM_HEAD=4
CNN_KERNEL_SIZE=7 # CNN_KERNEL_SIZE - 1
ATT_CACHE_OUTPUT_SIZE=128 # D_MODEL / NUM_HEAD * 2
# Triton specific parameters
# For more details, refer to
# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
MAX_DELAY=0
INSTANCE_NUM=2
INSTANCE_NUM_FOR_SCORING=2
MAX_BATCH_SIZE=16
MAX_BATCH_FOR_SCORING=16

model_repo_path=./model_repo_stateful_cuda_decoder

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
   echo "export to onnx files"
   wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2++_conformer_exp.tar.gz --no-check-certificate
   tar zxvf 20210601_u2++_conformer_exp.tar.gz
   model_dir=$(pwd)/20210601_u2++_conformer_exp

   cd ../../../
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 wenet/bin/export_onnx_gpu.py \
           --config=$model_dir/train.yaml \
           --checkpoint=$model_dir/final.pt \
           --cmvn_file=$model_dir/global_cmvn \
           --ctc_weight=0.5 \
           --output_onnx_dir=$onnx_model_dir \
           --streaming \
           --return_ctc_logprobs \
           --fp16 || exit 1
   cp $model_dir/units.txt $model_dir/train.yaml $onnx_model_dir
   cd -
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "auto gen config.pbtxt"
     dirs="encoder feature_extractor scoring streaming_wenet"
     if [ ! -d $model_repo_path ]; then
        echo "Please cd to model_repo_path"
        exit 1
     fi

     for dir in $dirs
     do
          cp $model_repo_path/$dir/config_template.pbtxt $model_repo_path/$dir/config.pbtxt

          sed -i "s/VOCAB_SIZE/${VOCAB_SIZE}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/MAX_DELAY/${MAX_DELAY}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/D_MODEL/${D_MODEL}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/CACHE_SIZE/${CACHE_SIZE}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/NUM_HEAD/${NUM_HEAD}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/CNN_KERNEL_SIZE/${CNN_KERNEL_SIZE}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/ATT_CACHE_OUTPUT_SIZE/${ATT_CACHE_OUTPUT_SIZE}/g" $model_repo_path/$dir/config.pbtxt

          if [ "$dir" == "decoder" ]; then
               sed -i "s/MAX_BATCH/${MAX_BATCH_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM}/g" $model_repo_path/$dir/config.pbtxt
          elif [ "$dir" == "scoring" ]; then
               sed -i "s/MAX_BATCH/${MAX_BATCH_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/DECODING_METHOD/${DECODING_METHOD}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
          else
               sed -i "s/MAX_BATCH/${MAX_BATCH_SIZE}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM}/g" $model_repo_path/$dir/config.pbtxt
          fi
     done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     echo "build tlg"
     # take aishell1 as example, you may build it using your own lm.
     # bash build_tlg.sh
     # tlg_dir=./data/lang_test
     # or you can download our pre-built TLG for this aishell1 tutorial.
     apt-get install git-lfs
     git-lfs install
     git clone https://huggingface.co/yuekai/aishell1_tlg_essentials.git
     cd aishell1_tlg_essentials
     git lfs pull
     cd -
     tlg_dir=./aishell1_tlg_essentials/output

     # mv TLG files to model_repo_path
     cp $tlg_dir/TLG.fst $model_repo_path/scoring/1/lang
     cp $tlg_dir/words.txt $model_repo_path/scoring/1/lang
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     echo "prepare files, you could skip it if you do it manually"
     mkdir -p $model_repo_path/encoder/1/
     cp $onnx_model_dir/encoder_fp16.onnx $model_repo_path/encoder/1/

     cp $onnx_model_dir/units.txt $model_repo_path/scoring/units.txt

     mkdir -p $model_repo_path/streaming_wenet/1/
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   echo "launch triton server"
   tritonserver --model-repository $model_repo_path \
               --pinned-memory-pool-byte-size=512000000 \
               --cuda-memory-pool-byte-size=0:1024000000 \
               --http-port 10086
fi
