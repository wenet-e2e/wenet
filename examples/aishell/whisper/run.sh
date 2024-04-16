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

stage=0
stop_stage=1

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard

train_set=train
# Optional train_config
# 1. Standard whisper largev3
#        train_config=conf/finetune_whisper_largev3.yaml
#        checkpoint=exp/whisper/large-v3/wenet_whisper.init-ctc.pt
# 2. Whisper largev3 with randomly init conv2d4
#        train_config=conf/finetune_whisper_largev3_conv2d4.yaml
#        checkpoint=exp/whisper/large-v3/wenet_whisper.remove-subsample.init-ctc.pt
train_config=conf/finetune_whisper_largev3.yaml
checkpoint=exp/whisper/large-v3/wenet_whisper.init-ctc.pt
dir=exp/finetune_whisper_largev3_conv1d2
tensorboard_dir=tensorboard
num_workers=8
prefetch=10

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=5
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_device=0
decoding_chunk_size=-1
ctc_weight=0.3
reverse_weight=0.0
decode_batch=4

train_engine=deepspeed

# model+optimizer or model_only, model+optimizer is more time-efficient but
# consumes more space, while model_only is the opposite
deepspeed_config=conf/ds_stage1.json
deepspeed_save_states="model+optimizer"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    for subdir in $(find "$dir" -maxdepth 1 -type d | grep -v "^$dir$")
    do
      # NOTE(xcsong): zero_to_fp32.py is automatically generated by deepspeed
      tag=$(basename "$subdir")
      echo "$tag"
      python3 ${dir}/zero_to_fp32.py \
        ${dir} ${dir}/${tag}.pt -t ${tag}
      rm -rf ${dir}/${tag}
    done
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
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
  base=$(basename $decode_checkpoint)
  result_dir=$dir/${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}
  mkdir -p ${result_dir}
  python wenet/bin/recognize.py --gpu ${decode_device} \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/test/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size ${decode_batch} \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $result_dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $result_dir/$mode/text > $result_dir/$mode/wer
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi
