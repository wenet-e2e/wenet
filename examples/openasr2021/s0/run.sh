#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Kai Tang).
# Apach 2.0

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# data
data=data
data_url=www.openslr.org/resources/33
nj=4

#langid: 101 Cantonese , 302 Kazakh , 401 mongolian
langs="101"
recog="101"

token_type=char
# bpemode (unigram or bpe)
nbpe=4500
bpemode=unigram

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

if [ "${token_type}" = bpe ]; then
    dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
    bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
elif [ "${token_type}" = char ]; then
    dict=data/lang_char/lang_char.txt
    bpe_model=
else
    echo "Error: not supported token_type"
    exit 0
fi

train_set=train_sp
train_dev=dev
recog_set=eval_$recog

# pretrained w2v-conformer encoder
enc_init=pretrain/conformer.pt
#reinit last pretrained encoder layer: https://arxiv.org/pdf/2107.04734.pdf
enc_init_mods='encoder.encoders.0,encoder.encoders.1,encoder.encoders.2,encoder.encoders.3,encoder.encoders.4,encoder.encoders.5,encoder.encoders.6,encoder.encoders.7,encoder.encoders.8,encoder.encoders.9,encoder.encoders.10,encoder.encoders.11,encoder.encoders.12,encoder.encoders.13,encoder.encoders.14,encoder.encoders.15,encoder.encoders.16,encoder.encoders.17,encoder.encoders.18,encoder.encoders.19,encoder.encoders.20,encoder.encoders.21,encoder.encoders.22,encoder.embed'

train_config=conf/train_conformer_large_10h.yaml
checkpoint=
cmvn=false
dir=exp/${langs}_finetune_10h

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=35

. utils/parse_options.sh || exit 1;

#Babel style data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Setting up individual languages"
  ./local/setup_languages.sh --langs "${langs}" --recog "${recog}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Data preparation
    for x in ${train_set} ${train_dev} ${recog_set}; do
        # Remove the space in text
        if [ "${token_type}" = char ]; then
            cp data/${x}/text data/${x}/text.org
            paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
            rm data/${x}/text.org
        fi
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)

    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    if [ "${token_type}" = bpe ]; then
        # we borrowed these code and scripts which are related bpe from ESPnet.
        cut -f 2- -d" " data/${train_set}/text | sort  > data/lang_char/input.txt
        tools/spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        tools/spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    elif [ "${token_type}" = char ]; then
        tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    fi

    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 1: format scp "
  #dumps such pipe-style-wav to real audio file
  for x in ${train_set} ${train_dev} ${recog_set}; do
    cp data/${x}/wav.scp data/${x}/wav.scp.org
    bash local/dump_wav.sh --nj 26 data/$x/wav.scp.org data/$x/segments data/$x/wav.scp
    rm  data/$x/wav.scp.org
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare data, prepare requried format"
  # For wav feature, just copy the data. mfcc/fbank extraction is done in training
  for x in ${train_set} ${train_dev} ${recog_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py  data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --data_type $data_type \
            --symbol_table $dict \
            ${bpemodel:+--bpe_model ${bpemodel}.model} \
            --train_data data/$train_set/data.list \
            --cv_data data/$train_dev/data.list \
            ${checkpoint:+--checkpoint $checkpoint} \
            ${enc_init:+--enc_init $enc_init} \
            --enc_init_mods $enc_init_mods \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $num_gpus \
            --ddp.rank $i \
            --ddp.dist_backend $dist_backend \
            --num_workers 6 \
            $cmvn_opts
    } &
    done
    wait
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
    # TODO, Add model average here
    mkdir -p $dir/test
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        python  wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --val_best
    fi
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    for mode in ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring; do
    for rtask in ${recog_set}; do
    {
        test_dir=$dir/test_${rtask}_${mode}
        mkdir -p $test_dir
        python  wenet/bin/recognize.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --data_type $data_type \
            --test_data data/$rtask/data.list \
            --checkpoint $decode_checkpoint \
            --beam_size 5 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            ${bpemodel:+--bpe_model ${bpemodel}.model} \
            --ctc_weight $ctc_weight \
            --result_file $test_dir/text_ori \
            $cmvn_opts \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
         if [ "${token_type}" = bpe ]; then
            tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_ori | sed -e "s/▁/ /g" > $test_dir/text
            python tools/compute-wer.py --char=0 --v=1 \
            data/$rtask/text $test_dir/text > $test_dir/wer
         elif [ "${token_type}" = char ]; then
            python tools/compute-wer.py --char=1 --v=1 \
            data/$rtask/text $test_dir/text_ori > $test_dir/wer
         fi
    } &
    done
    done
    wait

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    # Export the best model you want
    python wenet/bin/export_jit.py \
        --config $dir/train.yaml \
        --checkpoint $dir/avg_${average_num}.pt \
        --output_file $dir/final.zip
fi

