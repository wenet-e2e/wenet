#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
#           2022 burkliu(boji123@aliyun.com)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"

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

# modify this to your AISHELL-2 data path
# Note: the evaluation data (dev & test) is available at AISHELL.
# Please download it from http://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST%26DEV%20DATA.zip
train_set=/cfs/share/corpus/aishell-2/AISHELL-2/iOS/data
dev_set=/cfs/share/corpus/aishell-2/AISHELL-DEV-TEST-SET/iOS/dev
test_set=/cfs/share/corpus/aishell-2/AISHELL-DEV-TEST-SET/iOS/test

nj=16
dict=data/dict/lang_char.txt

train_set=train
train_config=conf/conformer_u2pp_rnnt.yaml
cmvn=true
dir=exp/`basename ${train_config%.*}`
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="rnnt_beam_search"

# Specify decoding_chunk_size if it's a unified dynamic chunk trained model
# -1 for full chunk
decoding_chunk_size=-1
# only used in rescore mode for weighting different scores
rescore_ctc_weight=0.5
rescore_transducer_weight=0.5
rescore_attn_weight=0.5
# only used in beam search, either pure beam search mode OR beam search inside rescoring
search_ctc_weight=0.3
search_transducer_weight=0.7

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/prepare_data.sh ${train_set} data/local/${train_set} data/${train_set} || exit 1;
    local/prepare_data.sh ${dev_set} data/local/dev data/dev || exit 1;
    local/prepare_data.sh ${test_set} data/local/test data/test || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # remove the space between the text labels for Mandarin dataset
    for x in ${train_set} dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 data/${x}/text.org) <(cut -f 2- data/${x}/text.org \
             | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done

    tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn data/$train_set/global_cmvn

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Prepare wenet required data
    echo "Prepare data, prepare required format"
    for x in dev test ${train_set}; do
        tools/make_raw_list.py data/$x/wav.scp data/$x/text data/$x/data.list
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    # You had better rm it manually before you start run.sh on first node.
    # rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="gloo"
    #dist_backend="nccl"
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    cmvn_opts=
    $cmvn && cp data/${train_set}/global_cmvn $dir
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
            --data_type raw \
            --symbol_table $dict \
            --train_data data/$train_set/data.list \
            --cv_data data/dev/data.list \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 4 \
            $cmvn_opts \
            2>&1 | tee -a $dir/train.log || exit 1;
    } &
    done
    wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        python wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --val_best \
            2>&1 | tee -a $dir/average.log || exit 1;
    fi

    for mode in ${decode_modes}; do
    {
        test_dir=$dir/test_${mode}_chunk_${decoding_chunk_size}
        mkdir -p $test_dir
        python wenet/bin/recognize.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --data_type raw \
            --test_data data/test/data.list \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $rescore_ctc_weight \
            --transducer_weight $rescore_transducer_weight \
            --attn_weight $rescore_attn_weight \
            --search_ctc_weight $search_ctc_weight \
            --search_transducer_weight $search_transducer_weight \
            --result_file $test_dir/text \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
         python tools/compute-wer.py --char=1 --v=1 \
            data/test/text $test_dir/text > $test_dir/wer
    } &
    done
    wait
fi
