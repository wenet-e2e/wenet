#!/bin/bash

# Copyright 2021 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
stage=1 # start from 0 if you need to start from data preparation
stop_stage=5

# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0

# data
# use your own data path, you can contact gigaspeech@speechcolab.orgfor getting data for data information about gigaspeech
# the preparation of gigaspeech dataset for wenet can be found https://github.com/SpeechColab/GigaSpeech
giga_data_dir=/mnt/nfs/ptm3/open-data/GigaSpeech
# gigaspeech training set
set=XL
train_set=train_`echo $set |tr 'A-Z' 'a-z'`
train_dev=dev
recog_set=test
# wav data dir
wave_data=data
nj=16
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_conformer.yaml
checkpoint=
cmvn=false
do_delta=false
dir=exp/sp_spec_aug

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=3
decode_modes="attention_rescoring ctc_greedy_search"

. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/gigaspeech_data_prep.sh --train-subset $set --stage 1 $giga_data_dir $wave_data
    sed -i "s/\t/ /g" $wave_data/gigaspeech_${train_set}/text
    sed -i "s/\t/ /g" $wave_data/gigaspeech_${train_dev}/text
    sed -i "s/\t/ /g" $wave_data/gigaspeech_${recog_set}/text
    for x in $train_dev $train_set $recog_set; do
        paste -d " " <(cut -f1 -d " " $wave_data/gigaspeech_$x/text) <(cut -f1 -d " " $wave_data/gigaspeech_$x/text) > $wave_data/gigaspeech_$x/spk2utt
    cp $wave_data/gigaspeech_$x/spk2utt $wave_data/gigaspeech_$x/utt2spk
        tools/fix_data_dir.sh $wave_data/gigaspeech_$x
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    echo "stage 1: generate segmented wav.scp and compute cmvn"
    if [ ! -f $wave_data/gigaspeech_$train_set/segments ]; then
        echo "$0: No such file segments" && exit 1;
    else
        # generate segmented wav.scp
        # the segmented wav.scp format is:
        # POD1000000004_S0000000 /GigaSpeech/audio/podcast/P0000/POD1000000004.opus,0.0,10.197
        # 0.0 is start time, 10.197 is end time (second)
        for x in $train_dev $train_set $recog_set; do
            mv $wave_data/gigaspeech_$x/wav.scp $wave_data/gigaspeech_$x/wav.scp.ori
            python tools/segment.py --segments $wave_data/gigaspeech_$x/segments \
                --input $wave_data/gigaspeech_$x/wav.scp.ori \
                 --output $wave_data/gigaspeech_$x/wav.scp
        done
    fi

    # optional
    # compute cmvn, perhaps you can sample some segmented examples fron wav.scp for cmvn computation
    python tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
        --in_scp $wave_data/gigaspeech_$train_set/wav.scp \
        --out_cmvn $wave_data/gigaspeech_$train_set/global_cmvn

fi


dict=$wave_data/lang_char_$set/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char_$set/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p $wave_data/lang_char_$set/

    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    # we borrowed these code and scripts which are related bpe from ESPnet.
    cut -f 2- -d" " $wave_data/gigaspeech_${train_set}/text > $wave_data/lang_char_$set/input.txt
    tools/spm_train --input=$wave_data/lang_char_$set/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char_$set/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    wc -l ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in $train_dev; do
        tools/format_data.sh --nj ${nj} \
            --feat-type opus --feat $wave_data/gigaspeech_$x/wav.scp --bpecode ${bpemodel}.model --segment true \
            $wave_data/gigaspeech_$x ${dict} > $wave_data/gigaspeech_$x/format.data

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
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    cmvn_opts=
    $cmvn && cp ${feat_dir}/${train_set}/global_cmvn $dir
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
            --train_data $wave_data/gigaspeech_$train_set/format.data \
            --cv_data $wave_data/gigaspeech_$train_dev/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 32 \
            $cmvn_opts \
            --use_amp
    } &
    done
    wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
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
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    # Polling GPU id begin with index 0
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    idx=0
    for test in $recog_set; do
        for mode in ${decode_modes}; do
        {
            {
                test_dir=$dir/${test}_${mode}
                mkdir -p $test_dir
                gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
                python wenet/bin/recognize.py --gpu $gpu_id \
                    --mode $mode \
                    --config $dir/train.yaml \
                    --test_data $wave_data/gigaspeech_$test/format.data \
                    --checkpoint $decode_checkpoint \
                    --beam_size 20 \
                    --batch_size 1 \
                    --penalty 0.0 \
                    --dict $dict \
                    --result_file $test_dir/text_bpe \
                    --ctc_weight $ctc_weight \
                    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

                cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
                cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp

                tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value
                paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value > $test_dir/text
                # a raw version wer without refining processs
                python tools/compute-wer.py --char=1 --v=1 \
                    $wave_data/gigaspeech_$test/text $test_dir/text > $test_dir/wer

                # for gigaspeech scoring
                cat $test_dir/text_bpe_key_tmp | sed -e "s/^/(/g" | sed -e "s/$/)/g" > $test_dir/hyp_key
                paste -d " " $test_dir/text_value $test_dir/hyp_key > $test_dir/hyp
                paste -d " " <(cut -f2- -d " " $wave_data/gigaspeech_$test/text) <(cut -f1 -d " " $wave_data/gigaspeech_$test/text | sed -e "s/^/(/g" | sed -e "s/$/)/g") > $wave_data/gigaspeech_$test/ref
                local/gigaspeech_scoring.py $wave_data/gigaspeech_$test/ref $test_dir/hyp $test_dir
            } &

            ((idx+=1))
            if [ $idx -eq $num_gpus ]; then
              idx=0
            fi
        }
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

