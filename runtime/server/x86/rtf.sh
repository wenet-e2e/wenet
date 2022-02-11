#!/bin/bash
# Copyright [2022-02-09] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
export GLOG_logtostderr=1
export GLOG_v=2

# 100 wavs.
wav_scp=./build/wav/wav.100.scp
# u2++ conformer, bi-rescore.
model_dir=./build/model/test
./build/decoder_main \
    --chunk_size 16 \
    --num_left_chunks -1 \
    --num_bins 40 \
    --ctc_weight 0.3 \
    --reverse_weight 0.5 \
    --wav_scp $wav_scp \
    --result hyp.txt \
    --api 2 \
    --repeat 5 \
    --model_path $model_dir/final.zip \
    --dict_path $model_dir/words.txt 2>&1 | tee log.txt
