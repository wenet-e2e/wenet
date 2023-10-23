#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

stage=0 # start from 0 if you need to start from data preparation
stop_stage=6

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

num_utts_per_shard=1000
data_url=https://www.openslr.org/resources/111
data_source=/home/work_nfs5_ssd/yhliang/data/aishell4
# modify this to your AISHELL-4 data path

nj=16
dict=data/dict/lang_char.txt

train_set=aishell4_train
dev_set=aishell4_test
test_sets=aishell4_test

train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  local/download_and_untar.sh ${data_source} ${data_url} train_L
  local/download_and_untar.sh ${data_source} ${data_url} train_M
  local/download_and_untar.sh ${data_source} ${data_url} train_S
  local/download_and_untar.sh ${data_source} ${data_url} test
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/prepare_data.sh ${data_source} || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # remove the space between the text labels for Mandarin dataset
  for x in ${train_set} ${test_sets}; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -d " " -f 1 data/${x}/text.org) <(cut -d " " -f 2 data/${x}/text.org \
      | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") > data/${x}/text
    rm data/${x}/text.org
  done

  tools/compute_cmvn_stats.py --num_workers 32 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/$train_set/global_cmvn

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Make train dict
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1
  tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in $train_set ${test_sets}; do
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads 32 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text $(realpath data/$x/shards) data/$x/data.list
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  # You had better rm it manually before you start run.sh on first node.
  # rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  # The number of gpus runing on each node/machine
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type shard \
      --symbol_table $dict \
      --train_data data/$train_set/data.list \
      --cv_data data/${dev_set}/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num}
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  for test_set in ${test_sets}; do
  {
    test_dir=$dir/${test_set}
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type shard \
      --test_data data/${test_set}/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 32 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --result_dir $test_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        data/${test_set}/text $test_dir/$mode/text > $test_dir/$mode/wer
    done
  }
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

