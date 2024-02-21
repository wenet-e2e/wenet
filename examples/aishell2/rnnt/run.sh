#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
#           2022 burkliu(boji123@aliyun.com)

. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

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

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

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
    echo "<sos/eos> 2" >> $dict # <eos>
    tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
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
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    if [ ${train_engine} == "deepspeed" ]; then
      echo "$0: using deepspeed"
    else
      echo "$0: using torch ddp"
    fi
    echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
    torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
             --rdzv_id=2023 --rdzv_backend="c10d" \
      wenet/bin/train.py \
              --train_engine ${train_engine} \
              --config $train_config \
              --data_type raw \
              --train_data data/$train_set/data.list \
              --cv_data data/dev/data.list \
              ${checkpoint:+--checkpoint $checkpoint} \
              --model_dir $dir \
              --ddp.dist_backend $dist_backend \
              --num_workers 4 \
              --pin_memory \
              --deepspeed_config ${deepspeed_config} \
              --deepspeed.save_states ${deepspeed_save_states} \
              2>&1 | tee -a $dir/train.log || exit 1;
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
    python wenet/bin/recognize.py --gpu 0 \
        --modes $decode_modes \
        --config $dir/train.yaml \
        --data_type raw \
        --test_data data/test/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --blank_penalty 0.0 \
        --ctc_weight $rescore_ctc_weight \
        --transducer_weight $rescore_transducer_weight \
        --attn_weight $rescore_attn_weight \
        --search_ctc_weight $search_ctc_weight \
        --search_transducer_weight $search_transducer_weight \
        --result_dir $dir \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    for mode in ${decode_modes}; do
        python tools/compute-wer.py --char=1 --v=1 \
            data/test/text $dir/$mode/text > $dir/$mode/wer
    done
fi
