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
stop_stage=5

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!
data=/export/data/asr-data/OpenSLR/33/
data_url=www.openslr.org/resources/33

nj=16
dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
# 7. conf/train_u2++_conformer.yaml: U2++ lite conformer, must load a well
#    trained model, and freeze encoder module, otherwise there will be a
#    autograd error
train_config=conf/train_conformer.yaml
dir=exp/conformer
tensorboard_dir=tensorboard
checkpoint=
num_workers=8
prefetch=10

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

train_engine=torch_ddp

deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  local/download_and_untar.sh ${data} ${data_url} data_aishell
  local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/aishell_data_prep.sh ${data}/data_aishell/wav \
    ${data}/data_aishell/transcript
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # remove the space between the text labels for Mandarin dataset
  for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) \
      <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
      > data/${x}/text
    rm data/${x}/text.org
  done

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/$train_set/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  echo "<sos/eos> 2" >> $dict
  tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+2}' >> ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  for x in dev test ${train_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  # NOTE(xcsong): Both ddp & deepspeed can be launched by torchrun
  # NOTE(xcsong): To unify single-node & multi-node training, we add
  #               all related args. You should change `nnodes` &
  #               `rdzv_endpoint` for multi-node, see
  #               https://pytorch.org/docs/stable/elastic/run.html#usage
  #               https://github.com/wenet-e2e/wenet/pull/2055#issuecomment-1766055406
  #               `rdzv_id` - A user-defined id that uniquely identifies the worker group for a job.
  #                           This id is used by each node to join as a member of a particular worker group.
  #               `rdzv_endpoint` - The rendezvous backend endpoint; usually in form <host>:<port>.
  # NOTE(xcsong): In multi-node training, some clusters require special NCCL variables to set prior to training.
  #               For example: `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=enp` + `NCCL_DEBUG=INFO`
  #               without NCCL_IB_DISABLE=1
  #                   RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL Version xxx
  #               without NCCL_SOCKET_IFNAME=enp  (IFNAME could be get by `ifconfig`)
  #                   RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:xxx
  #               ref: https://github.com/google/jax/issues/13559#issuecomment-1343573764
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data data/$train_set/data.list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
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
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  reverse_weight=0.5
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
    --reverse_weight $reverse_weight \
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
  chunk_size=-1
  ./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    --dict_path data/lang_test/words.txt \
    data/test/wav.scp data/test/text $dir/final.zip \
    data/lang_test/units.txt $dir/lm_with_runtime
  # Please see $dir/lm_with_runtime for wer
fi

# Optionally, you can decode with k2 hlg
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  if [ ! -f data/local/lm/lm.arpa ]; then
    echo "Please run prepare dict and train lm in Stage 7" || exit 1;
  fi

  # 8.1 Build decoding HLG
  required="data/local/hlg/HLG.pt data/local/hlg/words.txt"
  for f in $required; do
    if [ ! -f $f ]; then
      tools/k2/make_hlg.sh data/local/dict/ data/local/lm/ data/local/hlg
      break
    fi
  done

  # 8.2 Decode using HLG
  decoding_chunk_size=
  lm_scale=0.7
  decoder_scale=0.1
  r_decoder_scale=0.7
  decode_modes="hlg_onebest hlg_rescore"
  python wenet/bin/recognize.py --gpu 0 \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/test/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 16 \
    --blank_penalty 0.0 \
    --dict $dict \
    --word data/local/hlg/words.txt \
    --hlg data/local/hlg/HLG.pt \
    --lm_scale $lm_scale \
    --decoder_scale $decoder_scale \
    --r_decoder_scale $r_decoder_scale \
    --result_dir $dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $dir/$mode/text > $dir/$mode/wer
  done
fi

# Optionally, you can train with LF-MMI using k2
# Based on 20210601_u2++_conformer_exp/final.pt, we train 50 epocs with 1e-5 lr
# and average 10 best models, achieve 4.11 cer with hlg decoding
# Actually, you can achieve even lower cer by tuning lm_scale/decoder_scale/r_decoder_scale
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  # 9.1 Build token level bigram fst for LF-MMI training
  tools/k2/prepare_mmi.sh data/train/ data/dev data/local/lfmmi

  # 9.2 Run LF-MMI training from stage 4, modify below args in train.yaml
  # model: k2_model
  # model_conf:
  #   lfmmi_dir data/local/lfmmi

  # 9.3 Run HLG decode from stage 8.2
fi
