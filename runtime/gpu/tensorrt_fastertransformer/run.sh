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
stop_stage=5

#<wenet_onnx_gpu_models>
onnx_model_dir=$(pwd)/aishell_onnx
#<your_output_dir>
outputs_dir=./exp1

# modify model parameters according to your own model
d_model=256
head_num=4
vocab_size=4233

# paramters for TRT engines
MIN_BATCH=1
OPT_BATCH=16
MAX_BATCH=16
MAX_BATCH_FOR_SCORING=16

ENC_MIN_LEN=16
ENC_OPT_LEN=512
ENC_MAX_LEN=2048
DEC_MIN_LEN=$(( ENC_MIN_LEN / 4))
DEC_OPT_LEN=$(( ENC_OPT_LEN / 4))
DEC_MAX_LEN=$(( ENC_MAX_LEN / 4))

MIN_HYPS_PAD=2
OPT_HYPS_PAD=20
MAX_HYPS_PAD=64

BEAM_SIZE=10 # Don't modify it

mkdir -p $outputs_dir

model_repo_path=./model_repo_ft

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
   echo "export to onnx files"
   wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20211025_conformer_exp.tar.gz --no-check-certificate
   tar zxvf 20211025_conformer_exp.tar.gz
   model_dir=$(pwd)/20211025_conformer_exp
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
           --decoder_fastertransformer || exit 1
   cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir
   cd -
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "extract weights and replace onnx with plugins"

     mkdir -p /weight/enc
     mkdir -p /weight/dec
     python3 extract_weights.py --input_onnx $onnx_model_dir/encoder.onnx --output_dir /weight/enc || exit 1
     python3 extract_weights.py --input_onnx $onnx_model_dir/decoder.onnx --output_dir /weight/dec || exit 1

     python3 replace_plugin.py --input_onnx $onnx_model_dir/encoder.onnx \
                               --d_model $d_model --head_num $head_num --vocab_size $vocab_size\
                               --output_onnx ${outputs_dir}/encoder_plugin.onnx || exit 1

     python3 replace_plugin.py --input_onnx $onnx_model_dir/decoder.onnx \
                               --output_onnx ${outputs_dir}/decoder_plugin.onnx \
                               --d_model $d_model --head_num $head_num --vocab_size $vocab_size \
                               --num_layer 6 || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     echo "compile FasterTransformer"
     ft_path=./FasterTransformer
     pushd ${ft_path}

     export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
     CUR_DIR=`pwd`
     mkdir -p build
     cd build

     cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_VERBOSE_MAKEFILE=OFF \
          -DCMAKE_INSTALL_PREFIX=${CUR_DIR}/install \
          -DBUILD_TF=OFF \
          -DBUILD_PYT=OFF \
          -DBUILD_MULTI_GPU=OFF \
          -DUSE_NVTX=OFF \
          -DBUILD_EXAMPLE=ON \
          -DBUILD_TEST=OFF \
          -DBUILD_TRT=ON \
          -DBUILD_ORGIN_NET=OFF \
          ..

     make -j$(nproc) || exit 1
     popd
     cp ${ft_path}/build/lib/libtrt_wenet.so $outputs_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     cd $outputs_dir

     if [ ! -d /weight/enc ] || [ ! -d /weight/dec ]; then
        echo "Please extract weights and move them here first"
        exit 1
     fi

     echo "convert to trt"
     ${trtexec} \
          --onnx=./encoder_plugin.onnx \
          --minShapes=speech:${MIN_BATCH}x${ENC_MIN_LEN}x80,speech_lengths:${MIN_BATCH} \
          --optShapes=speech:${OPT_BATCH}x${ENC_OPT_LEN}x80,speech_lengths:${OPT_BATCH} \
          --maxShapes=speech:${MAX_BATCH}x${ENC_MAX_LEN}x80,speech_lengths:${MAX_BATCH} \
          --fp16 \
          --plugins=./libtrt_wenet.so \
          --saveEngine=./encoder.plan

     ${trtexec}   \
          --onnx=./decoder_plugin.onnx \
          --minShapes=encoder_out:${MIN_BATCH}x${DEC_MIN_LEN}x$d_model,encoder_out_lens:${MIN_BATCH},hyps_pad_sos_eos:${MIN_BATCH}x${BEAM_SIZE}x${MIN_HYPS_PAD},hyps_lens_sos:${MIN_BATCH}x${BEAM_SIZE},ctc_score:${MIN_BATCH}x${BEAM_SIZE} \
          --optShapes=encoder_out:${OPT_BATCH}x${DEC_OPT_LEN}x$d_model,encoder_out_lens:${OPT_BATCH},hyps_pad_sos_eos:${OPT_BATCH}x${BEAM_SIZE}x${OPT_HYPS_PAD},hyps_lens_sos:${OPT_BATCH}x${BEAM_SIZE},ctc_score:${OPT_BATCH}x${BEAM_SIZE} \
          --maxShapes=encoder_out:${MAX_BATCH}x${DEC_MAX_LEN}x$d_model,encoder_out_lens:${MAX_BATCH},hyps_pad_sos_eos:${MAX_BATCH}x${BEAM_SIZE}x${MAX_HYPS_PAD},hyps_lens_sos:${MAX_BATCH}x${BEAM_SIZE},ctc_score:${MAX_BATCH}x${BEAM_SIZE} \
          --fp16 \
          --plugins=./libtrt_wenet.so \
          --saveEngine=./decoder.plan \
          --buildOnly
          # infer with random input would cause illegal memory access error
     cd -
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
     echo "auto gen config.pbtxt"
     dirs="encoder decoder feature_extractor scoring attention_rescoring"
     DICT_PATH=$onnx_model_dir/words.txt
     VOCAB_SIZE=$vocab_size
     MAX_DELAY=0
     MAX_BATCH_SIZE=$MAX_BATCH
     D_MODEL=$d_model
     INSTANCE_NUM=1
     INSTANCE_NUM_FOR_SCORING=2

     if [ ! -d $model_repo_path ]; then
        echo "Please cd to model_repo_path"
        exit 1
     fi

     for dir in $dirs
     do
          cp $model_repo_path/$dir/config.pbtxt.template $model_repo_path/$dir/config.pbtxt

          sed -i "s|DICT_PATH|${DICT_PATH}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/BEAM_SIZE/${BEAM_SIZE}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/VOCAB_SIZE/${VOCAB_SIZE}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/MAX_DELAY/${MAX_DELAY}/g" $model_repo_path/$dir/config.pbtxt
          sed -i "s/D_MODEL/${D_MODEL}/g" $model_repo_path/$dir/config.pbtxt
          if [ "$dir" == "decoder" ]; then
               sed -i "s/MAX_BATCH/${MAX_BATCH_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM}/g" $model_repo_path/$dir/config.pbtxt
          elif [ "$dir" == "scoring" ]; then
               sed -i "s/MAX_BATCH/${MAX_BATCH_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM_FOR_SCORING}/g" $model_repo_path/$dir/config.pbtxt
          else
               sed -i "s/MAX_BATCH/${MAX_BATCH_SIZE}/g" $model_repo_path/$dir/config.pbtxt
               sed -i "s/INSTANCE_NUM/${INSTANCE_NUM}/g" $model_repo_path/$dir/config.pbtxt
          fi
     done

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     echo "prepare files, you could skip it if you do it manually"
     mkdir -p $model_repo_path/encoder/1/
     cp $outputs_dir/encoder.plan $model_repo_path/encoder/1/

     mkdir -p $model_repo_path/decoder/1/
     cp $outputs_dir/decoder.plan $model_repo_path/decoder/1/

     mkdir -p $model_repo_path/attention_rescoring/1/

     cp $outputs_dir/libtrt_wenet.so $model_repo_path/../
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
   echo "launch triton server"
   LD_PRELOAD=./libtrt_wenet.so tritonserver --model-repository $model_repo_path \
                                          --pinned-memory-pool-byte-size=512000000 \
                                          --cuda-memory-pool-byte-size=0:1024000000

fi