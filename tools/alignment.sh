#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0 # start from 0 if you need to start from data preparation
stop_stage=0

nj=16
feat_dir=raw_wav
dict=data/dict/lang_char.txt

dir=exp/
config=$dir/train.yaml
checkpoint=
checkpoint=/home/diwu/github/latest/wenet/examples/aishell/s0/exp/transformer/avg_20.pt
config=/home/diwu/github/latest/wenet/examples/aishell/s0/exp/transformer/train.yaml
set=
ali_format=$feat_dir/$set/format.data
ali_format=format.data
ali_result=$dir/ali

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    nj=32
    # Prepare required data for ctc alignment
    echo "Prepare data, prepare required format"
    for x in $set; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat $feat_dir/$x/wav.scp \
            $feat_dir/$x ${dict} > $feat_dir/$x/format.data.tmp

    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Test model, please specify the model you want to use by --checkpoint
        python wenet/bin/alignment_deprecated.py --gpu -1 \
            --config $config \
            --input_file $ali_format \
            --checkpoint $checkpoint \
            --batch_size 1 \
            --dict $dict \
            --result_file $ali_result \

fi


