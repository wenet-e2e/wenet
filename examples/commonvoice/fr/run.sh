#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2"
stage=0     # start from 0 if you need to start from data download
stop_stage=2

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

# data
download_path=/root/autodl-tmp
french_data=/root/autodl-tmp/cv-corpus-8.0-2022-01-19
# path to save preproecssed data
# export data=data
. ./path.sh
. ./tools/parse_options.sh || exit 1

nj=16

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
# 6. conf/train_u2++_conformer.yaml: U2++ conformer
# 7. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=
nbpe=5000

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=20
#decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_modes="attention attention_rescoring"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then

    echo "stage -1: Data download"
    echo "download Dataset!"
    local/download_data.sh ${download_path} ${french_data}
    echo "Finish stage 0"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    echo "stage 0: Data preparation"
    local/prepare_data.sh ${french_data}/fr
    echo "Finish stage 0"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: compute global cmvn"
    # compute cmvn
    python tools/compute_cmvn_stats.py --num_workers 1 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn data/${train_set}/global_cmvn
    echo "Finish stage 1"
fi


bpemode=unigram
dict=data/lang_char_/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char_/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char_/
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " data/${train_set}/text > data/lang_char_/input.txt
  tools/spm_train --input=data/lang_char_/input.txt --vocab_size=${nbpe} \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece \
    < data/lang_char_/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Prepare data, prepare required format"
  for x in dev test ${train_set}; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      python tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
  echo "Finish stage 3"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
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
      --train_data data/$train_set/data.list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
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
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/test_${mode}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu 0 \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type "raw" \
          --bpe_model $bpemodel.model \
          --test_data data/test/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 20 \
          --batch_size 1 \
          --penalty 0.0 \
          --dict $dict \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp

         tools/spm_decode --model=${bpemodel}.model --input_format=piece \
           < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value
        #sed -e "s/▁/ /g" $test_dir/text_bpe_value_tmp > $test_dir/text_value
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value > $test_dir/text
        # a raw version wer without refining processs
        python tools/compute-wer.py --char=1 --v=1 \
          data/test/text $test_dir/text > $test_dir/wer
      } &

      ((idx+=1))
      if [ $idx -eq $num_gpus ]; then
        idx=0
      fi
    }
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

