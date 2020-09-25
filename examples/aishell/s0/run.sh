#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0"

stage=4 # start from 0 if you need to start from data preparation
stop_stage=4
# data
data=/export/data/asr-data/OpenSLR/33/
data_url=www.openslr.org/resources/33

nj=16
feat_dir=fbank_pitch
dict=data/dict/lang_char.txt

train_set=train

train_config=conf/train_transformer.yaml
checkpoint=
cmvn=true
dir=exp/export

. utils/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    utils/perturb_data_dir_speed.sh 0.9 data/train data/train_sp0.9
    utils/perturb_data_dir_speed.sh 1.1 data/train data/train_sp1.1
    utils/combine_data.sh data/train_sp data/train data/train_sp0.9 data/train_sp1.1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Feature extraction
    mkdir -p $feat_dir
    for x in ${train_set} dev test; do
        cp -r data/$x $feat_dir
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj \
            --write_utt2num_frames true $feat_dir/$x
    done
    if $cmvn; then
        compute-cmvn-stats --binary=false scp:$feat_dir/$train_set/feats.scp \
            $feat_dir/$train_set/global_cmvn
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare data, prepare requried format"
    for x in dev test ${train_set}; do
        tools/format_data.sh --nj ${nj} --feat $feat_dir/$x/feats.scp \
            $feat_dir/$x ${dict} > $feat_dir/$x/format.data
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn ${feat_dir}/${train_set}/global_cmvn"
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data $feat_dir/$train_set/format.data \
            --cv_data $feat_dir/dev/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $num_gpus \
            --ddp.rank $i \
            --ddp.dist_backend $dist_backend \
            --num_workers 2 \
            $cmvn_opts
    } &
    done
    wait
fi

