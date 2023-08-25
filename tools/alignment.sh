#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0 # start from 0 if you need to start from data preparation
stop_stage=0

nj=16
dict=data/dict/lang_char.txt

dir=exp/
config=$dir/train.yaml
# model trained with trim tail will get a better alignment result
# (Todo) cif/attention/rnnt alignment
checkpoint=$dir/final.pt

set=test
ali_format=ali_format.data
ali_result=ali.res
blank_thres=0.9999
thres=0.00001
. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # Prepare required data for ctc alignment
    echo "Prepare data, prepare required format"
    for x in $set; do
        tools/make_raw_list.py data/$x/wav.scp data/$x/text \
          ali_format
    done
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Test model, please specify the model you want to use by --checkpoint
    mkdir -p exp_${thres}
    python wenet/bin/alignment.py --gpu -1 \
        --config $config \
        --input_file $ali_format \
        --checkpoint $checkpoint \
        --batch_size 1 \
        --dict $dict \
        --result_file $ali_result \
        --thres $thres \
        --blank_thres $blank_thres \
        --gen_praat

fi


