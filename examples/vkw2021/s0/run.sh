#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0
set -exo

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=-1 # start from 0 if you need to start from data preparation
stop_stage=0
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0
# data
data=data

nj=10
feat_dir=raw_wav
dict=data/dict/lang_char.txt
data_type=raw # raw or shard
num_utts_per_shard=1000

train_set=train
dev_set=combine_dev
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
name=vkw_bidirect_12conformer_hs2048_output256_att4_conv2d_char
train_config=conf/train_${name}.yaml #conf/train_12conformer_hs2048_output512_att4_conv2d_char.yaml
cmvn=true
dir=exp/train_${name}_new
checkpoint= #$dir/0.pt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # Data preparation
    $local/vkw_data_prep.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    x=dev_5h
    for z in lgv liv stv; do
        [ ! -f data/vkw/label/lab_${z}/${x}/wav_ori.scp ] && \
            mv data/vkw/label/lab_${z}/${x}/wav.scp \
                data/vkw/label/lab_${z}/${x}/wav_ori.scp && \
            cut -d" " -f -f 1,4 data/vkw/label/lab_${z}/${x}/wav_ori.scp |\
                > data/vkw/label/lab_${z}/${x}/wav.scp 
    done
    y=`echo $x | cut -d "_" -f 1`
    mkdir -p combine_${y}
    for f in text wav.scp segments; do
        for z in lgv liv stv; do
            cat data/vkw/label/lab_${z}/${x}/$f
        done > combine_${y}/$f
    done
    
    # remove the space between the text labels for Mandarin dataset
    # download and transfer to wav.scp
    for x in ${dev_set} ${train_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " \
            data/${x}/text.org | tr -d " ") > data/${x}/text
        rm data/${x}/text.org
    done
    #exit 0
fi
