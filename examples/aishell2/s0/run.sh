#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
stop_stage=6

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# modify this to your AISHELL-2 data path
# Note: the evaluation data (dev & test) is available at AISHELL.
# Please download it from http://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST%26DEV%20DATA.zip
trn_set=/mnt/nfs/ptm1/open-data/AISHELL-2/iOS/data
dev_set=/mnt/nfs/ptm1/open-data/AISHELL-DEV-TEST-SET/iOS/dev
tst_set=/mnt/nfs/ptm1/open-data/AISHELL-DEV-TEST-SET/iOS/test

nj=16
dict=data/dict/lang_char.txt

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
train_config=conf/train_unified_transformer.yaml
dir=exp/transformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/prepare_data.sh ${trn_set} data/local/${train_set} data/${train_set} || exit 1;
    local/prepare_data.sh ${dev_set} data/local/dev data/dev || exit 1;
    local/prepare_data.sh ${tst_set} data/local/test data/test || exit 1;
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
              --num_workers 2 \
              --pin_memory \
              --deepspeed_config ${deepspeed_config} \
              --deepspeed.save_states ${deepspeed_save_states}
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
    python wenet/bin/recognize.py --gpu 0 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 32 \
      --blank_penalty 0.0 \
      --ctc_weight $ctc_weight \
      --result_dir $dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        data/test/text $dir/$mode/text > $dir/$mode/wer
    done
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
  download_dir=data/local/DaCiDian
  git clone https://github.com/aishell-foundation/DaCiDian.git $download_dir
  mkdir -p data/local/dict
  cp $unit_file data/local/dict/units.txt
  tools/fst/prepare_dict.py $unit_file $download_dir/word_to_pinyin.txt \
      data/local/dict/lexicon.txt
  # 7.2 Segment text
  pip install jieba
  lm=data/local/lm
  mkdir -p $lm
  awk '{print $1}' data/local/dict/lexicon.txt | \
      awk '{print $1,99}' > $lm/word_seg_vocab.txt
  python local/word_segmentation.py $lm/word_seg_vocab.txt \
      data/train/text > $lm/text
  # 7.3 Train lm
  local/train_lms.sh
  # 7.4 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
      data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
  # 7.5 Decoding with runtime
  # reverse_weight only works for u2++ model and only left to right decoder is used when it is set to 0.0.
  reverse_weight=0.0
  chunk_size=-1
  ./tools/decode.sh --nj 16 --chunk_size $chunk_size\
      --beam 15.0 --lattice_beam 7.5 --max_active 7000 --blank_skip_thresh 0.98 \
      --ctc_weight 0.3 --rescoring_weight 1.0 --reverse_weight $reverse_weight\
      --fst_path data/lang_test/TLG.fst \
      --dict_path data/lang_test/words.txt \
      data/test/wav.scp data/test/text $dir/final.zip data/lang_test/units.txt \
      $dir/lm_with_runtime
  # See $dir/lm_with_runtime for wer
  tail $dir/lm_with_runtime/wer
fi
