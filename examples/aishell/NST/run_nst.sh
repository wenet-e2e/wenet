#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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


# This is an augmented version of aishell-1 "run.sh" to make the code compatible with noisy student training

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

stage=1 # start from 0 if you need to start from data preparation
stop_stage=8

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1


# here are extra parameters used in NST
cer_out_dir=""
dir=""
supervised_data_list=""
checkpoint=
unsupervised_data_list=""
data_list=""

hypo_name=""
out_data_list=""
#parameters with default values:
label=0
average_num=30
nj=16
num_split=1
cer_hypo_threshold=10
speak_rate_threshold=0
label_file="label.txt"
utter_time_file="utter_time.json"
enable_nst=1
job_num=0
dir_split="wenet_split_60_test/"
hypo_name="hypothesis_nst${job_num}.txt"
wav_dir="data/train/wenet_1k_untar/"
tar_dir="data/train/wenet_1khr_tar/"
untar_dir="data/train/wenet_1khr_untar/"
cer_hypo_dir="wenet_cer_hypo"
cer_label_dir="wenet_cer_label"
pseudo_data_ratio=0.75

dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard
num_utts_per_shard=1000
train_set=train
train_config=conf/train_conformer.yaml
average_checkpoint=true
target_pt=80
decode_checkpoint=$dir/$target_pt.pt

# here we only use attention_rescoring for NST
decode_modes="attention_rescoring"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

# print the settings
echo "setting for this run:"
echo "dir is ${dir}"
echo "data list is ${data_list}"
echo "job_num is ${job_num}"
echo "cer_out_dir is  ${cer_out_dir}"
echo "average_num is ${average_num}"
echo "checkpoint is ${checkpoint} "
echo "enable_nst is ${enable_nst} "

# we assumed that you have finished the data pre-process steps from -1 to 3 in aishell1/s0/run.sh .
# You can modify the "--train_data_supervised" to match your supervised data list.
# Here i used wenetspeech as the unsupervised data, you can run the data pre-process steps from -1 to 3 in
# wenetspeech/s0/run.sh ; you can modify "--train_data_supervised" to match your unsupervised data list.
# you can follow this process to generate your own dataset.
# I have also included my code for extracting data in local/...

# stage 1 is for training
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "********step 1 start time : $now ********"
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # the global_cmvn file need to be calculated by combining both supervised/unsupervised datasets,
  # and it should be positioned at data/${train_set}/global_cmvn .

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  echo "checkpoint is "  ${checkpoint}
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=2023 --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type $data_type \
      --train_data data/$train_set/$data_list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

# In stage 2, we get the averaged final checkpoint and calculate the test and dev accuracy
# please make sure your test and valid data.list are in the proper location.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # stage 5 we test with aishell dataset,
  echo "******** step 2 start time : $now ********"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi

  # export model
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  # test_wer
  for mode in ${decode_modes}; do
  {
    #test_dir=$dir/test_${mode}_${target_pt}pt  # for target pt
    test_dir=$dir/test_${mode}${average_num}pt   # for average pt
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --blank_penalty 0.0 \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "before compute-wer"
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text > $test_dir/wer
  } &
  done

#   dev_wer
  for mode in ${decode_modes}; do
  {
    #test_dir=$dir/test_${mode}_${target_pt}pt  # for target pt
    dev_dir=$dir/dev_${mode}${average_num}pt   # for average pt
    mkdir -p $dev_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/dev/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --blank_penalty 0.0 \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $dev_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "before compute-wer"
    python tools/compute-wer.py --char=1 --v=1 \
      data/dev/text $dev_dir/text > $dev_dir/wer
  } &
  done
  wait
fi


# split the (unsupervised) datalist into N sublists, where N depends on the number of available cpu in your cluster.
# when making inference, we compute N sublist in parallel.
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${enable_nst} -eq 0 ]; then
  echo "********step 3 start time : $now ********"
  python local/split_data_list.py \
    --job_nums $num_split \
    --data_list_path data/train/$unsupervised_data_list \
    --output_dir data/train/$dir_split

fi


# stage 4 will perform inference without language model on the given sublist(job num)
# here is example usages:
# bash run_nst.sh --stage 4 --stop-stage 4 --job_num $i --dir_split data/train/wenet_4khr_split_60/
# --hypo_name hypothesis_0.txt --dir exp/conformer_aishell2_wenet4k_nst4
# You need to specify the "job_num" n (n <= N), "dir_split" which is the dir path for split data
# "hypo_name" is the path for output hypothesis and "dir" is the path where we train and store the model.
# For each gpu, you can run with different job_num to perform data-wise parallel computing.
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "********step 4 start time : $now ********"
  # we assume you have run stage 2 so that avg_${average_num}.pt exists
  decode_checkpoint=$dir/avg_${average_num}.pt
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  mode="attention_rescoring"
  gpu_id=0
  echo "job number  ${job_num} "
  echo "data_list dir is  ${dir_split}"
  echo "hypo name is " $hypo_name
  echo "dir is ${dir}"

  python wenet/bin/recognize.py --gpu $gpu_id \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/train/${dir_split}data_sublist${job_num}/data_list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file data/train/${dir_split}data_sublist${job_num}/${hypo_name} \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    echo "end time : $now"

fi


# Generate wav.scp file and label.txt file(optional) for each sublist we generated in step 3.
# the wav_dir should be prepared in data processing step as we mentioned.
#You need to specify the "job_num" n (n <= N), "dir_split" which is the dir path for split data,
# "hypo_name" is the path for output hypothesis and "dir" is the path where we train and store the model.
# wav_dir is the directory that stores raw wav file and possible labels.
# if you have label for unsupervised dataset, set label = 1 other wise keep it 0
# For each gpu or cpu, you can run with different job_num to perform data-wise parallel computing.
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ ${enable_nst} -eq 0 ]; then
  echo "********step 5 start time : $now ********"
  python local/get_wav_labels.py \
    --dir_split data/train/${dir_split} \
    --hypo_name /$hypo_name \
    --wav_dir $wav_dir\
    --job_num $job_num \
    --label $label
fi

# Calculate cer-hypo between hypothesis with and without language model.
# We assumed that you have finished language model
# training using the wenet aishell-1 pipline. (You should have data/lang/words.txt , data/lang/TLG.fst files ready.)
# Here is an exmaple usage:
# bash run_nst.sh --stage 5 --stop-stage 5 --job_num n --dir_split data/train/wenet1k_redo_split_60/
# --cer_hypo_dir wenet1k_cer_hypo --hypo_name hypothesis_nst.txt --dir exp/conformer_no_filter_redo_nst6
# You need to specify the "job_num" n (n <= N), "dir_split" which is the dir path for split data
# "hypo_name" is the path for output hypothesis and "dir" is the path where we train and store the model.
# For each gpu, you can run with different job_num to perform data-wise parallel computing.
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "********step 6 start time : $now ********"
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "dir_split is " data/train/${dir_split}
  echo "nj is" $nj "hypo_file is" $hypo_name "cer out is" $cer_hypo_dir "lm is 4gram"
  echo "dir is " $dir
  if [ ! -f data/train/${dir_split}data_sublist${job_num}/${hypo_name}  ]; then
  echo "text file does not exists"
  exit 1;
  fi

  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    --dict_path data/lang_test/words.txt \
    data/train/${dir_split}data_sublist${job_num}/wav.scp \
    data/train/${dir_split}data_sublist${job_num}/${hypo_name} $dir/final.zip \
    data/lang_test/units.txt $dir/Hypo_LM_diff10/${cer_hypo_dir}_${job_num}
  now=$(date +"%T")
  echo "end time : $now"
fi

# (optional, only run this stage if you have true label for unsupervised data.)
# Calculate cer-label between true label and hypothesis with language model.
# You can use the output cer to evaluate NST's performance.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && [ ${label} -eq 1 ]; then
  echo "********step 7 start time : $now ********"
  chunk_size=-1
  mode="attention_rescoring"
  test_dir=$dir/test_${mode}_${job_num}
  now=$(date +"%T")
  echo "start time : $now"
  echo "GPU dir is " $job_num "dir_split is " data/train/${dir_split}
  echo "nj is" $nj "label_file is" $label_file "cer out is" $cer_label_dir "lm is 4gram"
  echo "dir is " $dir
  echo "label_file " data/train/${dir_split}data_sublist${job_num}/${label_file}
  if [ ! -f data/train/${dir_split}data_sublist${job_num}/${label_file}  ]; then
  echo "text file does not exists"
  exit 1;
  fi

  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    --dict_path data/lang_test/words.txt \
    data/train/${dir_split}data_sublist${job_num}/wav.scp \
    data/train/${dir_split}data_sublist${job_num}/${label_file} $dir/final.zip \
    data/lang_test/units.txt $dir/Hypo_LM_diff10/${cer_label_dir}_${job_num}
  now=$(date +"%T")
  echo "end time : $now"
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "********step 8 start time : $now ********"
  python local/generate_filtered_pseudo_label.py  \
    --cer_hypo_dir $cer_hypo_dir \
    --untar_dir data/train/$untar_dir \
    --wav_dir $wav_dir \
    --dir_num $job_num \
    --cer_hypo_threshold $cer_hypo_threshold \
    --speak_rate_threshold $speak_rate_threshold \
    --dir $dir \
    --tar_dir data/train/$tar_dir \
    --utter_time_file $utter_time_file

  python local/generate_data_list.py  \
    --tar_dir data/train/$tar_dir \
    --out_data_list data/train/$out_data_list \
    --supervised_data_list data/train/$supervised_data_list \
    --pseudo_data_ratio $pseudo_data_ratio

fi



