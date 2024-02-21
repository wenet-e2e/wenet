#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

nj=16
feat_dir=raw_wav

data_type=raw # raw or shard
num_utts_per_shard=1000

data_cat=legacy

train_set=train
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram


# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  local/download_data.sh # make soft link by yourself if you already have the dataset
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/prepare_data.sh $data_cat
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 \
      data/${dset}.orig data/${dset}
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  mkdir -p $feat_dir
  for x in ${train_set} dev test; do
    cp -r data/$x $feat_dir
  done
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn $feat_dir/$train_set/global_cmvn

fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
  tools/spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} \
    --model_type=${bpemode} --model_prefix=${bpemodel} \
    --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model \
    --output_format=piece < data/lang_char/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  if [ ! -f $feat_dir/$train_set/segments ]; then
    echo "$0: No such file segments" && exit 1;
  else
  for x in dev test ${train_set}; do
    tools/make_raw_list.py --segments $feat_dir/$x/segments \
    $feat_dir/$x/wav.scp $feat_dir/$x/text $feat_dir/$x/data.list
  done
  fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  # The number of gpus runing on each node/machine
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  cmvn_opts=
  $cmvn && cp ${feat_dir}/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=2023 --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --bpe_model $bpemodel.model \
      --train_data $feat_dir/$train_set/data.list \
      --cv_data $feat_dir/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.dist_backend $dist_backend \
      --num_workers 8 \
      $cmvn_opts \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data $feat_dir/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --blank_penalty 0.0 \
      --dict $dict \
      --bpe_model $bpemodel.model \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    python tools/compute-wer.py --char=1 --v=1 \
      $feat_dir/test/text $test_dir/text > $test_dir/wer
  } &
  done
  wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
