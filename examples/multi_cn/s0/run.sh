#!/bin/bash

# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# data
# data=/export/expts4/chaoyang/
dbase=
aidatatang_url=www.openslr.org/resources/62
aishell_url=www.openslr.org/resources/33
magicdata_url=www.openslr.org/resources/68
primewords_url=www.openslr.org/resources/47
stcmds_url=www.openslr.org/resources/38
thchs_url=www.openslr.org/resources/18

nj=16
feat_dir=raw_wav
dict=data/dict/lang_char.txt

train_set=train
dev_set=dev
test_sets="aishell aidatatang magicdata thchs aishell2 tal_asr"
has_aishell2=false  # AISHELL2 train set is not publically downloadable
                    # with this option true, the script assumes you have it in $dbase

# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
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
    # download all training data
    local/aidatatang_download_and_untar.sh $dbase/aidatatang $aidatatang_url aidatatang_200zh || exit 1;
    local/aishell_download_and_untar.sh $dbase/aishell $aishell_url data_aishell || exit 1;
    local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url train_set || exit 1;
    local/primewords_download_and_untar.sh $dbase/primewords $primewords_url || exit 1;
    local/stcmds_download_and_untar.sh $dbase/stcmds $stcmds_url || exit 1;
    local/thchs_download_and_untar.sh $dbase/thchs $thchs_url data_thchs30 || exit 1;

    # download all test data
    local/thchs_download_and_untar.sh $dbase/thchs $thchs_url test-noise || exit 1;
    local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url dev_set || exit 1;
    local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url test_set || exit 1;
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/aidatatang_data_prep.sh $dbase/aidatatang/aidatatang_200zh data/aidatatang || exit 1;
    local/aishell_data_prep.sh $dbase/aishell/data_aishell data/aishell || exit 1;
    local/thchs-30_data_prep.sh $dbase/thchs/data_thchs30 data/thchs || exit 1;
    local/magicdata_data_prep.sh $dbase/magicdata/ data/magicdata || exit 1;
    local/primewords_data_prep.sh $dbase/primewords data/primewords || exit 1;
    local/stcmds_data_prep.sh $dbase/stcmds data/stcmds || exit 1;
    local/tal_data_prep.sh $dbase/TAL/TAL_ASR-1/aisolution_data data/tal_asr || exit 1;
    local/tal_csasr_data_prep.sh $dbase/TAL/TAL_CSASR data/tal_csasr || exit 1;
    if $has_aishell2; then
        local/aishell2_data_prep.sh $dbase/aishell2/IOS data/aishell2/train || exit 1;
        local/aishell2_data_prep.sh $dbase/aishell2/IOS/dev data/aishell2/dev || exit 1;
        local/aishell2_data_prep.sh $dbase/aishell2/IOS/test data/aishell2/test || exit 1;
    fi

    tools/combine_data.sh data/${train_set} \
        data/{aidatatang,aishell,magicdata,primewords,stcmds,thchs,aishell2,tal_asr,tal_csasr}/train || exit 1;
    tools/combine_data.sh data/${dev_set} \
        data/{aidatatang,aishell,magicdata,thchs,aishell2,tal_asr}/dev || exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # For wav feature, just copy the data. Fbank extraction is done in training
    mkdir -p $feat_dir

    for x in ${train_set} ${dev_set}; do
        cp -r data/$x $feat_dir
    done

    for x in ${test_sets}; do
        cp -r data/$x/test $feat_dir/test_${x}
    done

    tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn $feat_dir/$train_set/global_cmvn

fi

bpecode=../../librispeech/s0/data/lang_char/train_960_unigram5000.model
trans_type=zh_char_en_bpe
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 -m ${bpecode} data/${train_set}/text --trans_type ${trans_type} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' \
        | grep -v '·' | grep -v '“' | grep -v "”" | grep -v "\[" | grep -v "\]" | grep -v "…" \
        | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    nj=32
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    feat_test_sets=""
    for x in ${test_sets}; do
        feat_test_sets=${feat_test_sets}" "test_${x}
    done
    for x in ${dev_set} ${train_set} ${feat_test_sets}; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat $feat_dir/$x/wav.scp \
            --bpecode ${bpecode} --trans_type ${trans_type} \
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
    rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    echo "$num_gpus"
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    cmvn_opts=
    $cmvn && cp ${feat_dir}/${train_set}/global_cmvn $dir
    $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data $feat_dir/$train_set/format.data \
            --cv_data $feat_dir/$dev_set/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $num_gpus \
            --ddp.rank $i \
            --ddp.dist_backend $dist_backend \
            --num_workers 2 \
            $cmvn_opts
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
    decoding_chunk_size=16
    ctc_weight=0.5
    idx=0
    for mode in ${decode_modes}; do
    {
        for x in ${test_sets}; do
        {
            test_dir=$dir/test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}/${x}
            mkdir -p $test_dir
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
            python wenet/bin/recognize.py --gpu $gpu_id \
                --mode $mode \
                --config $dir/train.yaml \
                --test_data $feat_dir/test_${x}/format.data \
                --checkpoint $decode_checkpoint \
                --beam_size 10 \
                --batch_size 1 \
                --penalty 0.0 \
                --dict $dict \
                --ctc_weight $ctc_weight \
                --result_file $test_dir/text \
                ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
            python tools/compute-wer.py --char=1 --v=1 \
                $feat_dir/test_${x}/text $test_dir/text > $test_dir/wer
        }
        done
    } &
    ((idx+=1))
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

