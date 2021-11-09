#!/bin/bash

# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
# Copyright 2021 Mobvoi Inc. All Rights Reserved. (Di Wu)
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=6

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0

# data
dbase=/ssd/nfs06/di.wu/open_source
aidatatang_url=www.openslr.org/resources/62
aishell_url=www.openslr.org/resources/33
magicdata_url=www.openslr.org/resources/68
primewords_url=www.openslr.org/resources/47
stcmds_url=www.openslr.org/resources/38
thchs_url=www.openslr.org/resources/18

nj=16

train_set=train
dev_set=dev

has_aishell2=false  # AISHELL2 train set is not publically downloadable
                    # With this option true, the script assumes you have it in
                    # $dbase
has_tal=false       # TAL data need download from Baidu SkyDrive
                    # With this option true, the script assumes you have
                    # TAL/TAL_ASR and TAL/TAL_ASR_mix in $dbase
data_type=raw # raw or shard
num_utts_per_shard=1000
shards_dir= # specify if you prefer to store to somewhere else
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
train_config=conf/train_conformer.yaml
# English modeling unit
# Optional 1. bpe 2. char
en_modeling_unit=bpe
dict=data/dict_$en_modeling_unit/lang_char.txt
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search"
decode_modes="$decode_modes attention attention_rescoring"

. tools/parse_options.sh || exit 1;

test_sets="aishell aidatatang magicdata thchs"
if $has_aishell2; then
  test_sets="$test_sets aishell2"
fi
if $has_tal; then
  test_sets="$test_sets tal_asr"
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  # download all training data
  local/aidatatang_download_and_untar.sh $dbase/aidatatang $aidatatang_url \
    aidatatang_200zh || exit 1;
  local/aishell_download_and_untar.sh $dbase/aishell $aishell_url \
    data_aishell || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url \
    train_set || exit 1;
  local/primewords_download_and_untar.sh $dbase/primewords $primewords_url \
    || exit 1;
  local/stcmds_download_and_untar.sh $dbase/stcmds $stcmds_url || exit 1;
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url data_thchs30 || \
    exit 1;

  # download all test data
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url test-noise \
    || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url \
    dev_set || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url \
    test_set || exit 1;
  # tal data need download from Baidu SkyDrive
  # AISHELL-2 database is free for academic research, not in the commerce,
  # if without permission.
  # You need to request the data from AISHELL company.
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/aidatatang_data_prep.sh $dbase/aidatatang/aidatatang_200zh \
    data/aidatatang || exit 1;
  local/aishell_data_prep.sh $dbase/aishell/data_aishell data/aishell \
    || exit 1;
  local/thchs-30_data_prep.sh $dbase/thchs/data_thchs30 data/thchs || exit 1;
  local/magicdata_data_prep.sh $dbase/magicdata/ data/magicdata || exit 1;
  local/primewords_data_prep.sh $dbase/primewords data/primewords || exit 1;
  local/stcmds_data_prep.sh $dbase/stcmds data/stcmds || exit 1;
  if $has_tal; then
    local/tal_data_prep.sh $dbase/TAL/TAL_ASR data/tal_asr || exit 1;
    local/tal_mix_data_prep.sh $dbase/TAL/TAL_ASR_mix data/tal_mix || exit 1;
  fi
  if $has_aishell2; then
    local/aishell2_data_prep.sh $dbase/aishell2/IOS data/aishell2/train \
      || exit 1;
    local/aishell2_data_prep.sh $dbase/aishell2/IOS/dev data/aishell2/dev \
      || exit 1;
    local/aishell2_data_prep.sh $dbase/aishell2/IOS/test data/aishell2/test \
      || exit 1;
  fi
  # Merge all data sets.
  train_sets=aidatatang,aishell,magicdata,primewords,stcmds,thchs
  dev_sets=aidatatang,aishell,magicdata,thchs
  if $has_aishell2; then
    train_sets=$train_sets,aishell2
    dev_sets=$dev_sets,aishell2
  fi
  if $has_tal; then
    train_sets=$train_sets,tal_asr,tal_mix
    dev_sets=$dev_sets,tal_asr
  fi
  unrolled_train_sets=$(eval echo data/{$train_sets}/train)
  unrolled_dev_sets=$(eval echo data/{$dev_sets}/dev)
  tools/combine_data.sh data/train $unrolled_train_sets || exit 1;
  tools/combine_data.sh data/dev $unrolled_dev_sets || exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  mkdir -p data_${en_modeling_unit}
  for x in ${train_set} ${dev_set}; do
      cp -r data/$x data_${en_modeling_unit}
  done

  for x in ${test_sets}; do
      cp -r data/$x/test data_${en_modeling_unit}/test_${x}
  done

  # Unified data format for char and bpe modelding
  # Here we use ▁ for blank among english words
  # Warning : it is "▁" symbol, not "_" symbol
  for x in train dev; do
    cp data_${en_modeling_unit}/${x}/text data_${en_modeling_unit}/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data_${en_modeling_unit}/${x}/text.org) \
      <(cut -f 2- -d" " data_${en_modeling_unit}/${x}/text.org \
      | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' \
      | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
      > data_${en_modeling_unit}/${x}/text
    sed -i 's/\xEF\xBB\xBF//' data_${en_modeling_unit}/${x}/text

  done

  for x in ${test_sets}; do
    cp data_${en_modeling_unit}/test_${x}/text \
      data_${en_modeling_unit}/test_${x}/text.org
    paste -d " " <(cut -f 1 -d" " data_${en_modeling_unit}/test_${x}/text.org) \
      <(cut -f 2- -d" " data_${en_modeling_unit}/test_${x}/text.org \
      | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' \
      | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
      > data_${en_modeling_unit}/test_${x}/text
    sed -i 's/\xEF\xBB\xBF//' data_${en_modeling_unit}/test_${x}/text
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some data to save time
  if $cmvn; then
    tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
      --in_scp data/${train_set}/wav.scp \
      --out_cmvn data_${en_modeling_unit}/$train_set/global_cmvn
  fi
fi

# This bpe model is trained on librispeech training data set.
bpecode=conf/train_960_unigram5000.model
trans_type_ops=
enable_bpe=
if [ $en_modeling_unit = "bpe" ]; then
  trans_type_ops="--trans_type cn_char_en_bpe"
  enable_bpe=true
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Make train dict
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  tools/text2token.py -s 1 -n 1 -m ${bpecode} \
    data_${en_modeling_unit}/${train_set}/text ${trans_type_ops} \
    | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' \
    | grep -v '·' | grep -v '“' | grep -v "”" | grep -v "\[" | grep -v "\]" \
    | grep -v "…" | awk '{print $0 " " NR+1}' >> ${dict}

  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare data, prepare requried format"
  feat_test_sets=""
  for x in ${test_sets}; do
    feat_test_sets=${feat_test_sets}" "test_${x}
  done
  for x in ${dev_set} ${train_set} ${feat_test_sets}; do
    if [ $data_type == "shard" ]; then
      sdir=${shards_dir:+$shards_dir/}shards_${en_modeling_unit}
      mkdir -p $sdir
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data_${en_modeling_unit}/$x/wav.scp \
        data_${en_modeling_unit}/$x/text $(realpath $sdir/$x) \
        data_${en_modeling_unit}/$x/data.list
    else
      tools/make_raw_list.py data_${en_modeling_unit}/$x/wav.scp \
        data_${en_modeling_unit}/$x/text data_${en_modeling_unit}/$x/data.list
    fi
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  # You had better rm it manually before you start run.sh on first node.
  # rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # The total number of processes/gpus, so that the master knows
  # how many workers to wait for.
  # More details about ddp can be found in
  # https://pytorch.org/tutorials/intermediate/dist_tuto.html
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data_${en_modeling_unit}/$train_set/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`

    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data data_${en_modeling_unit}/$train_set/data.list \
      --cv_data data_${en_modeling_unit}/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 4 \
      ${enable_bpe:+--bpe_model $bpecode} \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
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
  decoding_chunk_size=16
  ctc_weight=0.5
  idx=0
  for mode in ${decode_modes}; do
  {
    for x in ${test_sets}; do
    {
      test_name=test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}
      test_dir=$dir/$test_name/${x}
      mkdir -p $test_dir
      gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
      python wenet/bin/recognize.py --gpu $gpu_id \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type $data_type \
        --test_data data_${en_modeling_unit}/test_${x}/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --ctc_weight $ctc_weight \
        ${enable_bpe:+--bpe_model $bpecode} \
        --result_file $test_dir/text_${en_modeling_unit} \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

      cat $test_dir/text_${en_modeling_unit} | sed -e "s/▁/ /g" \
        > $test_dir/text
      cat data_${en_modeling_unit}/test_${x}/text | sed -e "s/▁/ /g" \
        > data_${en_modeling_unit}/test_${x}/text.tmp
      python tools/compute-wer.py --char=1 --v=1 \
        data_${en_modeling_unit}/test_${x}/text.tmp $test_dir/text \
        > $test_dir/wer
      rm data_${en_modeling_unit}/test_${x}/text.tmp
    }
    done
  } &
  ((idx+=1))
  done
  wait

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

