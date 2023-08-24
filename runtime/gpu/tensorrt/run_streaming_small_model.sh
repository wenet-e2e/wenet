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

trtexec=/usr/src/tensorrt/bin/trtexec
export CUDA_VISIBLE_DEVICES="0"
stage=-1
stop_stage=4

#<wenet_onnx_gpu_models>
onnx_model_dir=$(pwd)/u2pp_aishell2_onnx
#<your_output_dir>
outputs_dir=$(pwd)/exp_streaming_trt
model_repo_path=./model_repo_stateful_trt
mkdir -p $outputs_dir

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
   echo "export to onnx files"
   wget http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz --no-check-certificate
   tar zxvf 20210618_u2pp_conformer_exp.tar.gz
   model_dir=$(pwd)/20210618_u2pp_conformer_exp
   mkdir -p $onnx_model_dir
   cd ../../../
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 wenet/bin/export_onnx_gpu.py \
           --config=$model_dir/train.yaml \
           --checkpoint=$model_dir/final.pt \
           --cmvn_file=$model_dir/global_cmvn \
           --ctc_weight=0.5 \
           --output_onnx_dir=$onnx_model_dir \
           --fp16 \
           --streaming || exit 1
   cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir
   cd -
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    cd ./LayerNormPlugin
    make clean
    make
    cp LayerNorm.so $outputs_dir
    cd ..
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     echo "repalce onnx ops with layernorm plugin "
     polygraphy surgeon sanitize $onnx_model_dir/encoder_fp16.onnx --fold-constant -o $outputs_dir/encoderV2.onnx
     python3 replace_layernorm.py --input_onnx $outputs_dir/encoderV2.onnx \
                               --output_onnx $outputs_dir/encoderV3.onnx \
                               || exit 1
     polygraphy surgeon sanitize $outputs_dir/encoderV3.onnx --fold-constant -o $outputs_dir/encoderV4.onnx

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     echo "convert conformer encoder with layernorm plugin"
     MIN_BATCH=1
     OPT_BATCH=16
     MAX_BATCH=32
     $trtexec \
          --fp16 \
          --onnx=$outputs_dir/encoderV4.onnx \
          --minShapes=chunk_xs:${MIN_BATCH}x67x80,chunk_lens:${MIN_BATCH},offset:${MIN_BATCH}x1,att_cache:${MIN_BATCH}x12x4x80x128,cnn_cache:${MIN_BATCH}x12x256x7,cache_mask:${MIN_BATCH}x1x80 \
          --optShapes=chunk_xs:${OPT_BATCH}x67x80,chunk_lens:${OPT_BATCH},offset:${OPT_BATCH}x1,att_cache:${OPT_BATCH}x12x4x80x128,cnn_cache:${OPT_BATCH}x12x256x7,cache_mask:${OPT_BATCH}x1x80 \
          --maxShapes=chunk_xs:${MAX_BATCH}x67x80,chunk_lens:${MAX_BATCH},offset:${MAX_BATCH}x1,att_cache:${MAX_BATCH}x12x4x80x128,cnn_cache:${MAX_BATCH}x12x256x7,cache_mask:${MAX_BATCH}x1x80 \
          --plugins=$outputs_dir/LayerNorm.so \
          --saveEngine=$outputs_dir/encoder_fp16.plan
fi


# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#      echo "convert conformer encoder with layernorm plugin using multi-profile"
#      # profile1
#      MIN_BATCH1=9
#      OPT_BATCH1=16
#      MAX_BATCH1=32
#      # profile2
#      MIN_BATCH2=1
#      OPT_BATCH2=8
#      MAX_BATCH2=8

#      python3 export_streaming_conformer_trt.py \
#           --fp16 \
#           --onnxFile $outputs_dir/encoderV4.onnx \
#           --chunk_xs ${MIN_BATCH1}x67x80,${OPT_BATCH1}x67x80,${MAX_BATCH1}x67x80,${MIN_BATCH2}x67x80,${OPT_BATCH2}x67x80,${MAX_BATCH2}x67x80 \
#           --chunk_lens ${MIN_BATCH1},${OPT_BATCH1},${MAX_BATCH1},${MIN_BATCH2},${OPT_BATCH2},${MAX_BATCH2} \
#           --offset ${MIN_BATCH1}x1,${OPT_BATCH1}x1,${MAX_BATCH1}x1,${MIN_BATCH2}x1,${OPT_BATCH2}x1,${MAX_BATCH2}x1 \
#           --att_cache ${MIN_BATCH1}x12x4x80x128,${OPT_BATCH1}x12x4x80x128,${MAX_BATCH1}x12x4x80x128,${MIN_BATCH2}x12x4x80x128,${OPT_BATCH2}x12x4x80x128,${MAX_BATCH2}x12x4x80x128 \
#           --cnn_cache ${MIN_BATCH1}x12x256x7,${OPT_BATCH1}x12x256x7,${MAX_BATCH1}x12x256x7,${MIN_BATCH2}x12x256x7,${OPT_BATCH2}x12x256x7,${MAX_BATCH2}x12x256x7 \
#           --cache_mask ${MIN_BATCH1}x1x80,${OPT_BATCH1}x1x80,${MAX_BATCH1}x1x80,${MIN_BATCH2}x1x80,${OPT_BATCH2}x1x80,${MAX_BATCH2}x1x80 \
#           --plugin $outputs_dir/LayerNorm.so \
#           --trtFile $outputs_dir/encoder_fp16.plan \
#           --test || exit 1
# fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   echo "prepare model repo for triton"
   python3 ../scripts/convert.py --config=$onnx_model_dir/train.yaml --vocab=$onnx_model_dir/words.txt \
        --model_repo=$model_repo_path --onnx_model_dir=$onnx_model_dir

   cp $outputs_dir/encoder_fp16.plan $model_repo_path/encoder/1/
   cp $outputs_dir/LayerNorm.so $model_repo_path/../
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   echo "launch triton server"
   LD_PRELOAD=./LayerNorm.so tritonserver --model-repository $model_repo_path \
   --pinned-memory-pool-byte-size=2512000000 --cuda-memory-pool-byte-size=0:2024000000
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     echo "test conformer encoder with layernorm plugin using multi-profile"
     trtFile=$outputs_dir/encoder_fp16_multiprofile.plan
     python3 export_streaming_conformer_trt.py \
          --plugin $outputs_dir/LayerNorm.so \
          --trtFile $trtFile \
          --test

     echo "test single profile throughput"
     for B in 1 4 8 16 32
     do
          trtFile=$outputs_dir/encoder_fp16.plan
          /usr/src/tensorrt/bin/trtexec --fp16 --loadEngine=$trtFile --plugins=$outputs_dir/LayerNorm.so --noDataTransfers \
          --shapes=chunk_xs:${B}x67x80,chunk_lens:${B},offset:${B}x1,att_cache:${B}x12x4x80x128,cnn_cache:${B}x12x256x7,cache_mask:${B}x1x80 | grep qps
     done

     echo "test onnx throughput"
     python3 ../scripts/benchmark_onnx_throughput.py \
          --batch_sizes 1,4,8,16,32,64,128,256 \
          --onnxFile $onnx_model_dir/encoder_fp16.onnx \
          --log $outputs_dir/log.txt
fi
