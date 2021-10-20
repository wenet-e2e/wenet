#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0

set -exo
#conda activate wenet
#nvidia-smi -c 3

current_dir=$(pwd)
cd $current_dir
local=$current_dir/local

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0"

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=5 # start from 0 if you need to start from data preparation
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
data=$current_dir/data

nj=10
feat_dir=raw_wav
dict=data/dict/lang_char2.txt
data_type=raw # raw or shard
num_utts_per_shard=1000

train_set=train
dev_set=combine_dev
test_set=combine_test
finetune2_set=combine_finetune_5h
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
    for x in dev_5h; do
        for z in lgv liv stv; do
            [ ! -f data/vkw/label/lab_${z}/${x}/wav_ori.scp ] && \
                mv data/vkw/label/lab_${z}/${x}/wav.scp data/vkw/label/lab_${z}/${x}/wav_ori.scp && \
                sed "s/ffmpeg\ -i\ /ffmpeg\ -i\ data\/vkw\/data\/dat_${z}\//g" data/vkw/label/lab_${z}/${x}/wav_ori.scp | cut -d" " -f 1,4 > data/vkw/label/lab_${z}/${x}/wav.scp
        done
        #exit 0

        y=`echo $x | cut -d "_" -f 1`
        rm -rf data/combine_${y}
        [ ! -f data/combine_${y}/wav.scp ] && \
        $local/combine_data.sh data/combine_${y} \
            data/vkw/label/lab_lgv/${x} \
            data/vkw/label/lab_liv/${x} \
            data/vkw/label/lab_stv/${x}
    done

    # remove the space between the text labels for Mandarin dataset
    # download and transfer to wav.scp
    for x in ${dev_set} ${train_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
         
    done

    #exit 0

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: generate segmented wav.scp and compute cmvn"
    ## For wav feature, just copy the data. Fbank extraction is done in training
    mkdir -p $feat_dir
    for x in ${dev_set} ${train_set}; do
        mkdir -p $feat_dir/$x
        for f in spk2utt  text  utt2dur  utt2spk segments wav.scp; do
            cp -r data/$x/$f $feat_dir/$x/$f
        done
    done
    #exit 0

    for x in ${dev_set} ${train_set}; do
        [ ! -f $feat_dir/$x/wav.scp.ori ] && \
            mv $feat_dir/$x/wav.scp $feat_dir/$x/wav.scp.ori && \
            python tools/segment.py --segments $feat_dir/$x/segments \
                --input $feat_dir/$x/wav.scp.ori \
                --output $feat_dir/$x/wav.scp
    done

    ### generate global_cmvn using training set
    tools/compute_cmvn_stats.py --num_workers 48 --train_config $train_config \
        --in_scp $feat_dir/${train_set}/wav.scp \
        --out_cmvn $feat_dir/$train_set/global_cmvn

    #exit 0
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    echo "㕫 2" >> ${dict}
    echo "㖏 3" >> ${dict}

    tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | grep -P '[\p{Han}]' | awk '{print $0 " " NR+3}' >> ${dict}

    num_token=$(cat $dict | wc -l)
    echo "郎 $(expr $num_token)" >> $dict
    echo "凉 $(expr $num_token + 1)" >> $dict
    echo "氪 $(expr $num_token + 2)" >> $dict
    echo "宓 $(expr $num_token + 3)" >> $dict
    echo "OK $(expr $num_token + 4)" >> $dict
    echo "<sos/eos> $(expr $num_token + 5)" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare data, prepare requried format"
    for x in ${dev_set} ${train_set}; do
        if [ $data_type == "shard" ]; then
            tools/make_shard_list.py --resample 16000 --num_utts_per_shard $num_utts_per_shard \
                --num_threads 8 --segments $feat_dir/$x/segments $feat_dir/$x/wav.scp.ori $feat_dir/$x/text \
                $(realpath $feat_dir/$x/shards) $feat_dir/$x/data.list
        else
            tools/make_raw_list.py --segments $feat_dir/$x/segments $feat_dir/$x/wav.scp.ori $feat_dir/$x/text \
                $feat_dir/$x/data.list
        fi
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
        rank=$i ###`expr $node_rank \* $num_gpus + $i`
        echo "start training"
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --data_type $data_type \
            --symbol_table $dict \
            --train_data $feat_dir/$train_set/data.list \
            --cv_data $feat_dir/${dev_set}/data.list \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 4 \
            $cmvn_opts \
            --pin_memory
    }
    done
    wait
    exit 0
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        [ ! -f $decode_checkpoint ] && \
        python3 wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --val_best
    fi
    # Test model, please specify the model you want to use by --checkpoint
    # alignment input
    for sets in ${dev_set}; do
    #for sets in ${test_set}; do
        keywords_list=$data/vkw/keyword/kwlist
        ali_format=$feat_dir/${sets}/data.list
        checkpoint=$dir/1.pt #$dir/avg_${average_num}.pt
        keyword_results=$dir/keyword_results_${sets}
        ctc_results=$dir/ctc_results_${sets}
        python3 $local/vkw_kws_results.py --gpu 0 \
            --config $dir/train.yaml \
            --data_type $data_type \
            --symbol_table $dict \
            --num_workers 4 \
            --prefetch 32 \
            --input_data $feat_dir/${dev_set}/data.list \
            --checkpoint $checkpoint \
            --keyword_unit_dict $keywords_list \
            --keyword_results $keyword_results \
            --ctc_results $ctc_results
        exit 0

        new_dir=$(pwd)
        for y in "stv" "lgv" "liv"; do
            mkdir -p $dir/dev_${y}
            #[ ! -f $new_dir/data/vkw/score/dev_${y}/utter_map ] && \
            if [ $y == "lgv" ]; then
                grep "TV1" $keyword_results > $dir/dev_${y}/kws_results
                ./data/vkw/data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw/score/dev_${y}/ecf \
                    $new_dir/data/vkw/label/lab_${y}/dev_5h/segments \
                    $new_dir/data/vkw/score/dev_${y}/utter_map \
                    $dir/dev_${y}/kws_results \
                    $new_dir/data/vkw/keyword/kwlist.xml \
                    $new_dir/data/vkw/score/dev_${y}/rttm
                ./data/vkw/data/vkw/scripts/bin/F1.sh $dir/dev_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv
            elif [ $y == "liv" ]; then

                grep "sph_live" $keyword_results > $dir/dev_${y}/kws_results
                ./data/vkw/data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw/score/dev_${y}/ecf \
                    $new_dir/data/vkw/label/lab_${y}/dev_5h/segments \
                    $new_dir/data/vkw/score/dev_${y}/utter_map \
                    $dir/dev_${y}/kws_results \
                    $new_dir/data/vkw/keyword/kwlist.xml \
                    $new_dir/data/vkw/score/dev_${y}/rttm
                ./data/vkw/data/vkw/scripts/bin/F1.sh $dir/dev_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv

            elif [ $y == "stv" ]; then
                grep "sph_video" $keyword_results > $dir/dev_${y}/kws_results
                ./data/vkw/data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw/score/dev_${y}/ecf \
                    $new_dir/data/vkw/label/lab_${y}/dev_5h/segments \
                    $new_dir/data/vkw/score/dev_${y}/utter_map \
                    $dir/dev_${y}/kws_results \
                    $new_dir/data/vkw/keyword/kwlist.xml \
                    $new_dir/data/vkw/score/dev_${y}/rttm
                ./data/vkw/data/vkw/scripts/bin/F1.sh $dir/dev_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv
            else
                "invalid $y"
            fi
    done
    done
fi

