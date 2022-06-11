#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=0 # start from 0 if you need to start from data preparation
stop_stage=6
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0
# data
data=/export/data/asr-data/OpenSLR/33/
data_url=www.openslr.org/resources/33

nj=16
feat_dir=raw_wav
dict=data/dict/lang_char.txt

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
# 6. conf/train_u2++_conformer.yaml: U2++ conformer
# 7. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # remove the space between the text labels for Mandarin dataset
    for x in train dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done
    # For wav feature, just copy the data. Fbank extraction is done in training
    mkdir -p $feat_dir
    for x in ${train_set} dev test; do
        cp -r data/$x $feat_dir
    done

    tools/compute_cmvn_stats_deprecated.py --num_workers 16 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn $feat_dir/$train_set/global_cmvn

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    nj=32
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in dev test ${train_set}; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat $feat_dir/$x/wav.scp \
            $feat_dir/$x ${dict} > $feat_dir/$x/format.data.tmp

        tools/remove_longshortdata.py \
            --min_input_len 0.5 \
            --max_input_len 20 \
            --max_output_len 400 \
            --max_output_input_ratio 10.0 \
            --data_file $feat_dir/$x/format.data.tmp \
            --output_data_file $feat_dir/$x/format.data
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
        python wenet/bin/train_deprecated.py --gpu $gpu_id \
            --config $train_config \
            --train_data $feat_dir/$train_set/format.data \
            --cv_data $feat_dir/dev/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 2 \
            $cmvn_opts \
            --pin_memory
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
            --val_best
    fi
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    reverse_weight=0.0
    for mode in ${decode_modes}; do
    {
        test_dir=$dir/test_${mode}
        mkdir -p $test_dir
        python wenet/bin/recognize_deprecated.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --test_data $feat_dir/test/format.data \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $ctc_weight \
            --reverse_weight $reverse_weight \
            --result_file $test_dir/text \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
         python tools/compute-wer.py --char=1 --v=1 \
            $feat_dir/test/text $test_dir/text > $test_dir/wer
    } &
    done
    wait

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # Export the best model you want
    python wenet/bin/export_jit.py \
        --config $dir/train.yaml \
        --checkpoint $dir/avg_${average_num}.pt \
        --output_file $dir/final.zip \
        --output_quant_file $dir/final_quant.zip
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    # 7.1 Prepare dict
    unit_file=$dict
    mkdir -p data/local/dict
    cp $unit_file data/local/dict/units.txt
    tools/fst/prepare_dict.py $unit_file ${data}/resource_aishell/lexicon.txt \
        data/local/dict/lexicon.txt
    # 7.2 Train lm
    lm=data/local/lm
    mkdir -p $lm
    tools/filter_scp.pl data/train/text \
         $data/data_aishell/transcript/aishell_transcript_v0.8.txt > $lm/text
    local/aishell_train_lms.sh
    # 7.3 Build decoding TLG
    tools/fst/compile_lexicon_token_fst.sh \
        data/local/dict data/local/tmp data/local/lang
    tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
    # 7.4 Decoding with runtime
    # reverse_weight only works for u2++ model and only left to right decoder is used when it is set to 0.0.
    reverse_weight=0.0
    chunk_size=-1
    ./tools/decode.sh --nj 16 \
        --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
        --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
        --reverse_weight $reverse_weight --chunk_size $chunk_size \
        --fst_path data/lang_test/TLG.fst \
        --dict_path data/lang_test/words.txt \
        data/test/wav.scp data/test/text $dir/final.zip \
        data/lang_test/units.txt $dir/lm_with_runtime
    # See $dir/lm_with_runtime for wer
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    # Test model, please specify the model you want to use by --checkpoint
    # alignment input
    ali_format=$feat_dir/test/format.data
    # alignment output
    ali_result=$dir/ali
    python wenet/bin/alignment_deprecated.py --gpu -1 \
        --config $dir/train.yaml \
        --input_file $ali_format \
        --checkpoint $checkpoint \
        --batch_size 1 \
        --dict $dict \
        --result_file $ali_result \
        --gen_praat
fi

