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
stop_stage=6

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard
num_utts_per_shard=1000
data_url=https://www.openslr.org/resources/111
data_source=/bucket/output/jfs-hdfs/user/Archive/OpenSourceDataset/ASR/aishell4
# modify this to your AISHELL-4 data path

nj=16
dict=data/dict/lang_char.txt

train_set=aishell4_train
dev_set=aishell4_test
test_sets=aishell4_test

train_config=conf/train_conformer.yaml
dir=exp/conformer
tensorboard_dir=tensorboard
checkpoint=
num_workers=8
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_prefix_beam_search attention_rescoring ctc_greedy_search attention"

train_engine=torch_ddp

deepspeed_config=../../aishell/whisper/conf/ds_stage1.json
deepspeed_save_states="model_only"

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
  echo "<sos/eos> 2" >> $dict # <eos>
  tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in $train_set ${test_sets}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 32 --segments data/$x/segments \
        data/$x/wav.scp data/$x/text $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py --segments data/$x/segments \
        data/$x/wav.scp data/$x/text \
        data/$x/data.list.raw
    fi
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  # The number of gpus runing on each node/machine
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
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
      --data_type  $data_type \
      --train_data data/$train_set/data.list \
      --cv_data data/${dev_set}/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python3 wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=-1
  ctc_weight=0.5
  blank_penalty=0.0
  for test_set in ${test_sets}; do
  {
    test_dir=$dir/${test_set}_chunk${decoding_chunk_size}_ctc${ctc_weight}_bp${blank_penalty}
    mkdir -p $test_dir
    python3 wenet/bin/recognize.py --gpu 0 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type ${data_type} \
      --test_data data/${test_set}/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 32 \
      --blank_penalty ${blank_penalty} \
      --ctc_weight $ctc_weight \
      --result_dir $test_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    for mode in ${decode_modes}; do
      python3 tools/compute-wer.py --char=1 --v=1 \
        data/${test_set}/text $test_dir/$mode/text > $test_dir/$mode/wer
    done
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python3 wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

