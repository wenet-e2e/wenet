#!/bin/bash

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=4 # start from 0 if you need to start from data preparation
stop_stage=4

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

nj=16
feat_dir=raw_wav

data_type=raw
num_utts_per_shard=1000
prefetch=100

train_set=train_nodev
dev_set=train_dev

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
debug=false
num_workers=2
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search
              attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/hkust_data_prep.sh /mnt/cfs/database/hkust/LDC2005S15/ \
    /mnt/cfs/database/hkust/LDC2005T32/ || exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  mkdir -p ${feat_dir}_${en_modeling_unit}
  for x in ${train_set} ${dev_set}; do
    cp -r data/$x ${feat_dir}_${en_modeling_unit}
  done

  cp -r data/dev ${feat_dir}_${en_modeling_unit}/test

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn ${feat_dir}_${en_modeling_unit}/$train_set/global_cmvn

fi

# This bpe model is trained on librispeech training data set.
bpecode=conf/train_960_unigram5000.model
trans_type_ops=
bpe_ops=
if [ $en_modeling_unit = "bpe" ]; then
  trans_type_ops="--trans_type cn_char_en_bpe"
  bpe_ops="--bpecode ${bpecode}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Make train dict
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  paste -d " " \
    <(cut -f 1 -d" " ${feat_dir}_${en_modeling_unit}/${train_set}/text) \
    <(cut -f 2- -d" " ${feat_dir}_${en_modeling_unit}/${train_set}/text \
    | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' \
    | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " " ) \
    > ${feat_dir}_${en_modeling_unit}/${train_set}/text4dict
  sed -i 's/\xEF\xBB\xBF//' \
    ${feat_dir}_${en_modeling_unit}/${train_set}/text4dict

  tools/text2token.py -s 1 -n 1 -m ${bpecode} \
    ${feat_dir}_${en_modeling_unit}/${train_set}/text4dict ${trans_type_ops} \
    | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' \
    | grep -v '·' | grep -v '“' | grep -v "”" | grep -v "\[" | grep -v "\]" \
    | grep -v "…" \
    | awk '{print $0 " " NR+1}' >> ${dict}

  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet requried data
  echo "Prepare data, prepare requried format"
  for x in ${dev_set} ${train_set} test; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 ${feat_dir}_${en_modeling_unit}/$x/wav.scp \
        ${feat_dir}_${en_modeling_unit}/$x/text \
        $(realpath ${feat_dir}_${en_modeling_unit}/$x/shards) \
        ${feat_dir}_${en_modeling_unit}/$x/data.list
    else
      tools/make_raw_list.py ${feat_dir}_${en_modeling_unit}/$x/wav.scp \
      ${feat_dir}_${en_modeling_unit}/$x/text \
      ${feat_dir}_${en_modeling_unit}/$x/data.list
    fi
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
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  # The total number of processes/gpus, so that the master knows
  # how many workers to wait for.
  # More details about ddp can be found in
  # https://pytorch.org/tutorials/intermediate/dist_tuto.html
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp ${feat_dir}_${en_modeling_unit}/$train_set/global_cmvn $dir
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
      --prefetch $prefetch \
      --train_data ${feat_dir}_${en_modeling_unit}/$train_set/data.list \
      --cv_data ${feat_dir}_${en_modeling_unit}/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts \
      --pin_memory \
      --bpe_model ${bpecode}
  } &
  done
  wait
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
  decoding_chunk_size=-1
  ctc_weight=0.5
  idx=0
  for mode in ${decode_modes}; do
  {
    test_dir="$dir/"`
      `"test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}/test"
    mkdir -p $test_dir
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
    python wenet/bin/recognize.py --gpu $gpu_id \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data ${feat_dir}_${en_modeling_unit}/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --result_file $test_dir/text_${en_modeling_unit} \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    if [ $en_modeling_unit == "bpe" ]; then
      tools/spm_decode --model=${bpecode} --input_format=piece \
      < $test_dir/text_${en_modeling_unit} | sed -e "s/▁/ /g" > $test_dir/text
    else
      cat $test_dir/text_${en_modeling_unit} \
      | sed -e "s/▁/ /g" > $test_dir/text
    fi
    # Cer used to be consistent with kaldi & espnet
    python tools/compute-cer.py --char=1 --v=1 \
      ${feat_dir}_${en_modeling_unit}/test/text $test_dir/text > $test_dir/wer
  } &
  ((idx+=1))
  done
  wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

