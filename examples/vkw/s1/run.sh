#!/bin/bash
#set -exo
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apdcephfs/share_1157259/users/yougenyuan/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apdcephfs/share_1157259/users/yougenyuan/software/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/apdcephfs/share_1157259/users/yougenyuan/software/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/apdcephfs/share_1157259/users/yougenyuan/software/miniconda3/bin:$PATH"
    fi  
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate wenet
#nvidia-smi -c 3

current_dir=/apdcephfs/share_1157259/users/yougenyuan/backup/handover/wenet/examples/vkw/s1
cd $current_dir
local=$current_dir/local

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
stage=10 # start from 0 if you need to start from data preparation
stop_stage=10
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0
# data
data=$current_dir/data #/export/data/asr-data/OpenSLR/33/
#data_url=www.openslr.org/resources/33

nj=32
feat_dir=$data
dict=data/dict/lang_char2.txt

train_set=train_20210525_vkw_ddt_1kh_org_wuwFbank
dev_set=combine_dev
test_set=combine_test
finetune_set=combine_finetune
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
compress=true
fbank_conf=conf/fbank.conf
dir=exp/train_${name}_new
checkpoint= #$dir/0.pt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=5
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # Data preparation
    $local/vkw_data_prep.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for x in ${dev_set} ${test_set}; do
        y=`echo $x | cut -d "_" -f 1`
        if [ $x == "finetune_5h" ]; then
            y=$x
        fi
        rm -rf data/combine_${y}
        [ ! -f data/combine_${y}/wav.scp ] && \
        utils/combine_data.sh data/combine_${y} \
            data/vkw/label/lab_lgv/train_20210525_vkw_lgv_3kh_org_mfcchires/${x} \
            data/vkw/label/lab_liv/train_20210525_vkw_liv_500h_org_mfcchires/${x} \
            data/vkw/label/lab_stv/train_20210525_vkw_stv_500h_org_mfcchires/${x}
    done
    #exit 0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    #cut -f 2- -d" " data/${train_set}/text | tr " " "\n" \
    #    | sort | uniq | grep -a -v -e '^\s*$' |grep '^[a-z]' | grep '[0-9]$' | awk '{print $0 " " NR+1}' >> ${dict}

    tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}

    num_token=$(cat $dict | wc -l)
    echo "OK $num_token" >> $dict
    temp=`expr $num_token + 1`
    echo "<sos/eos> $temp" >> $dict # <eos>

    ### you need manually validate all the characters of keywords should be in the dict ###

    for x in ${dev_set} ${test_set}; do
        #[ ! -f data/${x}/text.org ] && \
            cp data/${x}/text data/${x}/text.org && \
            paste -d " " <(cut -d" " -f 1 data/${x}/text.org) <(cut -d" " -f 2- data/${x}/text.org | tr -d " ") > data/${x}/text
        #[ ! -f data/${x}/text_char ] && \
            python3 $local/map_text_to_char.py data/${x}/text $dict data/${x}/text_char
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in ${dev_set} ${test_set} ${train_set}; do
    #for x in ${train_set}; do
        #[ ! -f $feat_dir/$x/wav_ori.scp ] && \
        #    mv $feat_dir/$x/wav.scp $feat_dir/$x/wav_ori.scp
 
        [ ! -f $feat_dir/$x/feats.scp ] && \
            bash steps/make_fbank.sh --cmd "$train_cmd" --nj ${nj} \
                --write_utt2num_frames true --fbank_config $fbank_conf --compress $compress \
                $feat_dir/$x $feat_dir/$x/log $feat_dir/$x/mfcc

        if [ `wc -l $feat_dir/$x/feats.scp` -le `wc -l $feat_dir/$x/text` ]; then
            utils/fix_data_dir.sh $feat_dir/$x && \
            mv $feat_dir/$x/text_char $feat_dir/$x/.backup/text_char && \
            utils/filter_scp.pl $feat_dir/$x/text $feat_dir/$x/.backup/text_char > $feat_dir/$x/text_char
        fi

        #[ ! -f $feat_dir/$x/format.data.char.tmp ] && \
        bash $local/format_data_char.sh --nj ${nj} \
            --feat-type kaldi --feat $feat_dir/$x/feats.scp \
            $feat_dir/$x ${dict} > $feat_dir/$x/format.data.char.tmp

        rm -f $feat_dir/$x/format.data.char
        python3 tools/remove_longshortdata.py \
            --min_input_len 16 \
            --max_input_len 3999 \
            --min_output_len 2 \
            --max_output_len 500 \
            --min_output_input_ratio 0.001 \
            --max_output_input_ratio 0.125 \
            --data_file $feat_dir/$x/format.data.char.tmp \
            --output_data_file $feat_dir/$x/format.data.char
    done
    if $cmvn; then
        compute-cmvn-stats --binary=false scp:$feat_dir/$train_set/feats.scp \
            $feat_dir/$train_set/global_cmvn
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    #set -x
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    # You had better rm it manually before you start run.sh on first node.
    [ -f $INIT_FILE ] && rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=$num_gpus #`expr $num_gpus \* $num_nodes`
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
        rank=$i #`expr $node_rank \* $num_gpus + $i`

        python3 wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data $feat_dir/${train_set}/format.data.char \
            --cv_data $feat_dir/$dev_set/format.data.char \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 8 \
            $cmvn_opts \
            --pin_memory
    } &
    done
    wait
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    # Test model, please specify the model you want to use by --checkpoint
    # alignment input
    for sets in ${dev_set}; do
    #for sets in ${test_set}; do
        ali_format=$feat_dir/${sets}/format.data.char.tmp
        segments=$feat_dir/${sets}/segments
        checkpoint=$dir/avg_${average_num}.pt
        # alignment output
        rttm_result=$dir/rttm2_${sets}
        python3 $local/vkw_alignment.py --gpu -1 \
            --config $dir/train.yaml \
            --input_file $ali_format \
            --checkpoint $checkpoint \
            --batch_size 1 \
            --dict $dict \
            --segments $segments \
            --rttm_file $rttm_result
    done

fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
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
        ali_format=$feat_dir/${sets}/format.data.char.tmp
        checkpoint=$dir/avg_${average_num}.pt
        keyword_results=$dir/keyword_results_${sets}
        ctc_results=$dir/ctc_results_${sets}
        python3 $local/vkw_keyword_results.py --gpu -1 \
            --config $dir/train.yaml \
            --input_file $ali_format \
            --checkpoint $checkpoint \
            --batch_size 32 \
            --dict $dict \
            --keyword_unit_dict $data/dict/keywords.list \
            --model_unit char \
            --keyword_results $keyword_results \
            --ctc_results $ctc_results

        new_dir=/apdcephfs/share_1157259/users/yougenyuan/backup/wenet-b263ef0/examples/vkw/s1_handover
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

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
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
    for sets in ${test_set}; do
        ali_format=$feat_dir/${sets}/format.data.char.tmp
        checkpoint=$dir/avg_${average_num}.pt
        keyword_results=$dir/keyword_results_${sets}
        ctc_results=$dir/ctc_results_${sets}

        python3 $local/vkw_keyword_results.py --gpu -1 \
            --config $dir/train.yaml \
            --input_file $ali_format \
            --checkpoint $checkpoint \
            --batch_size 32 \
            --dict $dict \
            --keyword_unit_dict $data/dict/keywords_test.list \
            --model_unit char \
            --keyword_results $keyword_results \
            --ctc_results $ctc_results
        #exit 0

        new_dir=/apdcephfs/share_1157259/users/yougenyuan/backup/wenet-b263ef0/examples/vkw/s1_handover/baseline
        for y in "stv" "lgv" "liv"; do
        #for y in "lgv"; do
            mkdir -p $dir/test_${y}
            if [ $y == "lgv" ]; then
                grep "TV1" $keyword_results > $dir/test_${y}/kws_results
                ./data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw_testset/score/test_${y}/ecf \
                    $new_dir/data/vkw_testset/label/lab_${y}/test_20h/segments \
                    $new_dir/data/vkw_testset/score/test_${y}/utter_map \
                    $dir/test_${y}/kws_results \
                    $new_dir/data/vkw_testset/keyword/kwlist.xml \
                    $new_dir/data/vkw_testset/score/test_${y}/rttm
                ./data/vkw/scripts/bin/F1.sh $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv &>  $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/F1.txt
            elif [ $y == "liv" ]; then

                grep "sph_live" $keyword_results > $dir/test_${y}/kws_results
                ./data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw_testset/score/test_${y}/ecf \
                    $new_dir/data/vkw_testset/label/lab_${y}/test_20h/segments \
                    $new_dir/data/vkw_testset/score/test_${y}/utter_map \
                    $dir/test_${y}/kws_results \
                    $new_dir/data/vkw_testset/keyword/kwlist.xml \
                    $new_dir/data/vkw_testset/score/test_${y}/rttm
                ./data/vkw/scripts/bin/F1.sh $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv &>  $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/F1.txt

            elif [ $y == "stv" ]; then
                grep "sph_video" $keyword_results > $dir/test_${y}/kws_results
                ./data/vkw/scripts/bin/results_to_score.sh $new_dir/data/vkw_testset/score/test_${y}/ecf \
                    $new_dir/data/vkw_testset/label/lab_${y}/test_20h/segments \
                    $new_dir/data/vkw_testset/score/test_${y}/utter_map \
                    $dir/test_${y}/kws_results \
                    $new_dir/data/vkw_testset/keyword/kwlist.xml \
                    $new_dir/data/vkw_testset/score/test_${y}/rttm
                ./data/vkw/scripts/bin/F1.sh $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv &>  $dir/test_${y}/kws_outputs/f4de_scores_unnormalized/F1.txt
            else
                "invalid $y"
            fi
    done
    done
fi
