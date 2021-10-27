#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# data
data_url=www.openslr.org/resources/12
# use your own data path
datadir=/nfsa/diwu/open-dir
nj=16
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_conformer_large.yaml
checkpoint=
cmvn=true
do_delta=false

dir=exp/sp_spec_aug

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=20
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

. utils/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"
recog_set="test_clean"
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/dev_clean data/dev_other

    # compute global CMVN
    compute-cmvn-stats --binary=false scp:data/${train_set}/feats.scp data/${train_set}/global_cmvn

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
    tools/spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    tools/spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    wc -l ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in dev ${recog_set} ${train_set}; do
        tools/format_data.sh --nj ${nj} --feat data/$x/feats.scp --bpecode ${bpemodel}.model \
            data/$x ${dict} > data/$x/format.data.tmp

        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        tools/remove_longshortdata.py \
            --min_input_len 0.5 \
            --max_input_len 20 \
            --max_output_len 400 \
            --max_output_input_ratio 10.0 \
            --data_file data/$x/format.data.tmp \
            --output_data_file data/$x/format.data

    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
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
        python wenet/bin/train_deprecated.py --gpu $gpu_id \
            --config $train_config \
            --train_data data/$train_set/format.data \
            --cv_data data/dev/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $num_gpus \
            --ddp.rank $i \
            --ddp.dist_backend $dist_backend \
            --num_workers 1 \
            $cmvn_opts
    } &
    done
    wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
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
    # static dataloader is need for attention_rescoring decode
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
        test_dir=$dir/${test}_${mode}
        mkdir -p $test_dir
        python wenet/bin/recognize_deprecated.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --test_data data/$test/format.data \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --result_file $test_dir/text_bpe \
            --ctc_weight $ctc_weight \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
        tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_bpe | sed -e "s/▁/ /g" > $test_dir/text
        python tools/compute-wer.py --char=1 --v=1 \
            data/$test/text $test_dir/text > $test_dir/wer
    } &
    done
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

