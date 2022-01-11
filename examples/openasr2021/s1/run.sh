#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Kai Tang).
# Apach 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# data
data=data/train
data_url=www.openslr.org/resources/33

nj=4
feat_dir=mfcc

langs="302"
recog="302"

token_type=bpe
# bpemode (unigram or bpe)
nbpe=4500
bpemode=unigram

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
enc_init=pretrain/conformer_18.pt
#enc_init=
enc_init_mods="encoder."

train_config=conf/train_conformer_large_10h.yaml
checkpoint=
cmvn=false
dir=exp/${langs}_finetune_10h

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=35

. utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Setting up individual languages"
  ./local/setup_languages.sh --langs "${langs}" --recog "${recog}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Data preparation
    utils/perturb_data_dir_speed.sh 0.9 data/train data/train_sp0.9
    utils/perturb_data_dir_speed.sh 1.1 data/train data/train_sp1.1
    utils/combine_data.sh data/train_sp data/train data/train_sp0.9 data/train_sp1.1
    for x in ${train_set} ${train_dev} ${recog_set}; do
       sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
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
    # Feature extraction
    mkdir -p $feat_dir
    for x in ${train_set} ${train_dev} ${recog_set} ; do
        cp -r data/$x $feat_dir
        steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
           --mfcc-config conf/mfcc_hires.conf  --write_utt2num_frames true data/$x exp/make_mfcc/${x} ${feat_dir}/$x
    done
    if $cmvn; then
        compute-cmvn-stats --binary=false scp:data/$train_set/feats.scp \
            data/$train_set/global_cmvn
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
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


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in ${recog_set} ${train_dev}  ${train_set}; do
        tools/format_data.sh --nj ${nj} --feat data/$x/feats.scp \
            ${bpemodel:+--bpecode ${bpemodel}.model} \
            data/$x ${dict} > data/$x/format.data
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
    $cmvn && cmvn_opts="--cmvn ${feat_dir}/${train_set}/global_cmvn"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python wenet/bin/train_deprecated.py --gpu $gpu_id \
            --config $train_config \
            --train_data data/$train_set/format.data \
            --cv_data data/$train_dev/format.data \
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
    $cmvn && cmvn_opts="--cmvn ${feat_dir}/${train_set}/global_cmvn"
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
    for rtask in ${recog_test}; do
    {
        test_dir=$dir/test_${rtask}_${mode}
        mkdir -p $test_dir
        python  wenet/bin/recognize_deprecated.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --test_data data/$rtask/format.data \
            --checkpoint $decode_checkpoint \
            --beam_size 5 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $ctc_weight \
            --result_file $test_dir/text_ori \
            $cmvn_opts \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
         if [ "${token_type}" = bpe ]; then
            tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_ori | sed -e "s/▁/ /g" > $test_dir/text
            python2 tools/compute-wer.py --char=0 --v=1 \
            data/$rtask/text $test_dir/text > $test_dir/wer
         elif [ "${token_type}" = char ]; then
            python2 tools/compute-wer.py --char=1 --v=1 \
            data/$rtask/text $test_dir/text > $test_dir/wer
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

