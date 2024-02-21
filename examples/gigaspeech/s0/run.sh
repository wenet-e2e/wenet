#!/bin/bash

# Copyright 2021 Mobvoi Inc. All Rights Reserved.

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

# data
# use your own data path, you can contact gigaspeech@speechcolab.orgfor getting data for data information about gigaspeech
# the preparation of gigaspeech dataset for wenet can be found https://github.com/SpeechColab/GigaSpeech
giga_data_dir=/export/expts6/corpus/data/en-asr-data/16k/GigaSpeech
shards_dir=/ssd/nfs06/unified_data/giga_shards
# gigaspeech training set
set=XL
train_set=train_`echo $set |tr 'A-Z' 'a-z'`
train_dev=dev
recog_set=test
# wav data dir
data=data
nj=16
# Optional train_config
# 1. conf/train_transformer.yaml: Standard Conformer
# 2. conf/train_transformer_bidecoder.yaml: Bidecoder Conformer
train_config=conf/train_conformer_bidecoder.yaml
checkpoint=
do_delta=false
dir=exp/sp_spec_aug

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=3
decode_modes="attention_rescoring ctc_greedy_search"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  ### Task dependent. You have to make data the following preparation part by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 0: Data preparation"
  local/gigaspeech_data_prep.sh --train-subset $set --stage 1 $giga_data_dir $data
  sed -i "s/\t/ /g" $data/${train_set}/text
  sed -i "s/\t/ /g" $data/${train_dev}/text
  sed -i "s/\t/ /g" $data/${recog_set}/text
  for x in $train_dev $train_set $recog_set; do
    paste -d " " <(cut -f1 -d " " $data/$x/text) <(cut -f1 -d " " $data/$x/text) > $data/$x/spk2utt
    cp $data/$x/spk2utt $data/$x/utt2spk
    tools/fix_data_dir.sh $data/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  echo "stage 1: generate segmented wav.scp and compute cmvn"
  # the format of wav.segment.scp is:
  # POD1000000004_S0000000 /GigaSpeech/audio/podcast/P0000/POD1000000004.opus,0.0,10.197
  # 0.0 is start time, 10.197 is end time (second)
  for x in $train_dev $train_set $recog_set; do
    python tools/segment.py --segments $data/$x/segments \
      --input $data/$x/wav.scp \
      --output $data/$x/wav.segment.scp
  done

  # optional
  # compute cmvn, perhaps you can sample some segmented examples fron wav.scp for cmvn computation
  python tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $data/$train_set/wav.segment.scp \
    --out_cmvn $data/$train_set/global_cmvn
fi


dict=$data/lang_char_$set/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$data/lang_char_$set/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p $data/lang_char_$set/
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1
  echo "<sos/eos> 2" >> $dict # <eos>

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $data/${train_set}/text > $data/lang_char_$set/input.txt
  tools/spm_train --input=$data/lang_char_$set/input.txt --vocab_size=${nbpe} \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece \
    < $data/lang_char_$set/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'
  echo -e "It requires ${RED}1.2T ${NOCOLOR}space for $shards_dir, please make sure you have enough space"
  echo -e "It takes about ${RED}12 ${NOCOLOR}hours with 32 threads"

  for x in $train_dev $train_set $recog_set; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 1000 \
      --num_threads 32 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text \
      $(realpath $dst) data/$x/data.list
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
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
      --config $train_config \
      --data_type "shard" \
      --train_data $data/$train_set/data.list \
      --cv_data $data/$train_dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.dist_backend $dist_backend \
      --num_workers 16 \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # TODO, Add model average here
  mkdir -p $dir/test
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
  for test in $recog_set; do
  {
    result_dir=$dir/${test}
    python wenet/bin/recognize.py --gpu 0 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type "shard" \
      --test_data $data/$test/format.data \
      --checkpoint $decode_checkpoint \
      --beam_size 20 \
      --batch_size 32 \
      --blank_penalty 0.0 \
      --result_dir $result_dir \
      --ctc_weight $ctc_weight \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

    for mode in $decode_modes; do
      test_dir=$result_dir/$mode
      cp $test_dir/text $test_dir/text_bpe
      cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
      cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp

      tools/spm_decode --model=${bpemodel}.model --input_format=piece \
        < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value
      paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value > $test_dir/text
      # a raw version wer without refining processs
      python tools/compute-wer.py --char=1 --v=1 \
        $data/$test/text $test_dir/text > $test_dir/wer

      # for gigaspeech scoring
      cat $test_dir/text_bpe_key_tmp | sed -e "s/^/(/g" | sed -e "s/$/)/g" > $test_dir/hyp_key
      paste -d " " $test_dir/text_value $test_dir/hyp_key > $test_dir/hyp
      paste -d " " <(cut -f2- -d " " $data/$test/text) \
        <(cut -f1 -d " " $data/$test/text | \
        sed -e "s/^/(/g" | sed -e "s/$/)/g") > $data/$test/ref
      local/gigaspeech_scoring.py $data/$test/ref $test_dir/hyp $test_dir
    done
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi

