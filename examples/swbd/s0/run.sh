#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
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
data_type=shard # raw or shard
num_utts_per_shard=1000
prefetch=100
# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# data directory
swbd1_dir=/home/backup_nfs2/hlyu/swbd/LDC97S62
eval2000_dir="/home/backup_nfs2/hlyu/swbd/LDC2002S09/hub5e_00 /home/backup_nfs2/hlyu/swbd/LDC2002T43"

train_set=train_nodup
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/swbd1_data_download.sh ${swbd1_dir}
  local/swbd1_prepare_dict.sh
  local/swbd1_data_prep.sh ${swbd1_dir}
  local/eval2000_data_prep.sh ${eval2000_dir}
  # process the train set by
  # 1) convert lower to upper
  # 2) remove ._._ -1 symbols from text
  # 3) subset training set and dev set
  # 4) remove duplicated utterances
  cp data/train/text data/train/text.org
  paste -d" " <(cut -f 1 -d" " data/train/text.org) \
    <(cut -f 2- -d" " data/train/text.org | tr "[:lower:]" "[:upper:]") > data/train/text
  sed -i 's/\._/ /g; s/\.//g; s/THEM_1/THEM/g' data/train/text
  tools/subset_data_dir.sh --first data/train 4000 data/train_dev  # 5hr 6min
  n=$(($(wc -l < data/train/text) - 4000))
  tools/subset_data_dir.sh --last data/train ${n} data/train_nodev
  tools/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup
  # process eval2000 set by
  # 1) remove tags (%AH) (%HESITATION) (%UH)
  # 2) remove <B_ASIDE> <E_ASIDE>
  # 3) remove "(" or ")"
  # 4) remove file with empty text
  cp data/eval2000/text data/eval2000/text.org
  paste -d "" \
      <(cut -f 1 -d" " data/eval2000/text.org) \
      <(awk '{$1=""; print toupper($0)}' data/eval2000/text.org \
      | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' \
      | sed -e "s/(//g" -e "s/)//g") \
      | sed -e 's/\s\+/ /g' > data/eval2000/text.org2
  awk -F ' ' '{if(length($2) != 0) print $0}' data/eval2000/text.org2 > data/eval2000/text
  tools/fix_data_dir.sh data/eval2000
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  mkdir -p ${feat_dir}
  for x in ${train_set} train_dev eval2000; do
    cp -r data/${x} ${feat_dir}
  done
  tools/compute_cmvn_stats.py --num_workers 16 --train_config ${train_config} \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn ${feat_dir}/${train_set}/global_cmvn

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

  tools/spm_train --input=data/lang_char/input.txt \
    --vocab_size=${nbpe} \
    --character_coverage=1.0 \
    --model_type=${bpemode} \
    --model_prefix=${bpemodel} \
    --input_sentence_size=100000000 \
    --user_defined_symbols="[LAUGHTER],[NOISE],[VOCALIZED-NOISE]"
  tools/spm_encode --model=${bpemodel}.model \
    --output_format=piece < data/lang_char/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

  num_token=$(cat ${dict} | wc -l)
  echo "<sos/eos> ${num_token}" >> ${dict} # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare requried format"
  for x in ${train_set} train_dev eval2000; do
    if [ ${data_type} == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard ${num_utts_per_shard} \
        --num_threads ${nj} ${feat_dir}/${x}/wav.scp ${feat_dir}/${x}/text \
        $(realpath ${feat_dir}/${x}/shards) ${feat_dir}/${x}/data.list
    else
      tools/make_raw_list.py ${feat_dir}/${x}/wav.scp ${feat_dir}/${x}/text \
        ${feat_dir}/${x}/data.list
    fi
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p ${dir}
  INIT_FILE=${dir}/ddp_init
  # You had better rm it manually before you start run.sh on first node.
  # rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f ${INIT_FILE})
  echo "$0: init method is $init_method"
  # The number of gpus runing on each node/machine
  num_gpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # The total number of processes/gpus, so that the master knows
  # how many workers to wait for.
  # More details about ddp can be found in
  # https://pytorch.org/tutorials/intermediate/dist_tuto.html
  world_size=`expr ${num_gpus} \* ${num_nodes}`
  echo "total gpus is: ${world_size}"
  cmvn_opts=
  ${cmvn} && cp ${feat_dir}/${train_set}/global_cmvn ${dir}
  ${cmvn} && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < ${num_gpus}; ++i)); do
  {
    gpu_id=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr ${node_rank} \* ${num_gpus} + ${i}`
    python wenet/bin/train.py --gpu ${gpu_id} \
      --config ${train_config} \
      --data_type ${data_type} \
      --symbol_table ${dict} \
      --prefetch ${prefetch} \
      --bpe_model ${bpemodel}.model \
      --train_data ${feat_dir}/${train_set}/data.list \
      --cv_data ${feat_dir}/train_dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir ${dir} \
      --ddp.init_method ${init_method} \
      --ddp.world_size ${world_size} \
      --ddp.rank ${rank} \
      --ddp.dist_backend ${dist_backend} \
      --num_workers 4 \
      ${cmvn_opts} \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=${dir}/avg_${average_num}.pt
    echo "do model average and final checkpoint is ${decode_checkpoint}"
    python wenet/bin/average_model.py \
      --dst_model ${decode_checkpoint} \
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
    test_dir=${dir}/test_${mode}
    mkdir -p ${test_dir}
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data $feat_dir/eval2000/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --bpe_model $bpemodel.model \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    sed -i.bak -r 's/<blank> //g' ${test_dir}/text
    mv ${test_dir}/text ${test_dir}/text.bak2
    tools/spm_decode --model=${bpemodel}.model --input_format=piece \
        < ${test_dir}/text.bak2 | sed -e "s/▁/ /g" > ${test_dir}/text
    python tools/compute-wer.py --char=1 --v=1 \
      $feat_dir/eval2000/text $test_dir/text > $test_dir/wer
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
