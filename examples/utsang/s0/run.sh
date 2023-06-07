#!/bin/bash
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
#export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
#也可以在Linux命令行直接指定参数（运行阶段）
stage=4 # start from 0 if you need to start from data preparation
stop_stage=6

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0
# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!
data=/data1/zhang/tibetan #训练数据存放路径
data_url=www.openslr.org/resources/22 #这是维语数据下载地址（data_thuyg20.tar.gz是语音识别数据）

nj=16 #特征维度
dict=data/dict/lang_char.txt    #生成的字典存放目录

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
# data_type=shard
# num_utts_per_shard=1000
num_utts_per_shard=100 #每个shard文件存放音频个数，默认1000

train_set=train   #用训练集运行
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml  #训练配置
cmvn=true #Cepstral Mean and Variance Normalization; 倒谱均值 方差归一化
# dir=exp/conformer-kham-8head #实验结果目录，产生的模型文件和训练结果放在什么位置，比如可以改为exp/conformer-20220707-1
dir=exp/conformer-amdo-16head
checkpoint=       #训练中断后，可以指定checkpoint接着训练模型
num_workers=1
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=true   #对多个模型取平均，一般比 只取最后一个模型 效果更好
decode_checkpoint=$dir/final.pt
average_num=5            #用最后多少个模型取平均
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring" #四种解码方式

deepspeed=false # 不使用 DeepSpeed 框架进行模型训练
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"  # 表示只保存模型状态（推理时需要），而不是保存优化器和 AMP 状态。当使用 ModelParallel 或 Pipeline 并行时，每个 GPU 都有可能拥有其自己的优化器和 AMP 状态，因此将 deepspeed_save_states 设为 "all" 可以保存所有状态

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  local/download_and_untar.sh ${data} ${data_url} data_aishell
  local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  local/aishell_data_prep.sh ${data}/data_amdo/wav \
    ${data}/data_amdo/transcript
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
  tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
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
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init
  rm -f ${INIT_FILE}  # remove previous INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${deepspeed} == true ]; then
    echo "using deepspeed"
    # NOTE(xcsong): deepspeed fails with gloo, see
    #   https://github.com/microsoft/DeepSpeed/issues/2818
    dist_backend="nccl"
    [ ! -f data/$train_set/data.list.filter ] && \
      python tools/filter_uneven_data.py data/$train_set/data.list \
        $data_type $num_gpus $num_utts_per_shard data/$train_set/data.list.filter
    deepspeed --include localhost:$CUDA_VISIBLE_DEVICES \
      wenet/bin/train.py \
        --deepspeed \
        --deepspeed_config ${deepspeed_config} \
        --deepspeed.save_states ${deepspeed_save_states} \
        --ddp.dist_backend $dist_backend \
        --ddp.init_method $init_method \
        --data_type  $data_type \
        --config $train_config \
        --symbol_table  data/dict/lang_char.txt \
        --train_data data/$train_set/data.list.filter \
        --cv_data data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        $cmvn_opts \
        --pin_memory
  else
    echo "using torch ddp"
    for ((i = 0; i < $num_gpus; ++i)); do
    {
      gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
      # Rank of each gpu/process used for knowing whether it is
      # the master of a worker.
      rank=`expr $node_rank \* $num_gpus + $i`
      python wenet/bin/train.py --gpu $gpu_id \
        --config $train_config \
        --data_type $data_type \
        --symbol_table $dict \
        --train_data data/$train_set/data.list \
        --cv_data data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --ddp.init_method $init_method \
        --ddp.world_size $world_size \
        --ddp.rank $rank \
        --ddp.dist_backend $dist_backend \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        $cmvn_opts \
        --pin_memory
    } &
    done
    wait
  fi
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
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
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
      data/test/text $test_dir/text > $test_dir/wer
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
  tools/fst/prepare_dict.py $unit_file ${data}/resource_tibetan/lexicon.txt \
    data/local/dict/lexicon.txt
  # 7.2 Train lm
  lm=data/local/lm
  mkdir -p $lm
  tools/filter_scp.pl data/train/text \
    $data/data_amdo/transcript/aishell_transcript_v0.8.txt > $lm/text
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
  for mode in hlg_onebest hlg_rescore; do
  {
    test_dir=$dir/test_${mode}
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 16 \
      --penalty 0.0 \
      --dict $dict \
      --word data/local/hlg/words.txt \
      --hlg data/local/hlg/HLG.pt \
      --lm_scale $lm_scale \
      --decoder_scale $decoder_scale \
      --r_decoder_scale $r_decoder_scale \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text > $test_dir/wer
  }
  done
fi

# Optionally, you can train with LF-MMI using k2
# Based on 20210601_u2++_conformer_exp/final.pt, we train 50 epocs with 1e-5 lr
# and average 10 best models, achieve 4.11 cer with hlg decoding
# Actually, you can achieve even lower cer by tuning lm_scale/decoder_scale/r_decoder_scale
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  # 9.1 Build token level bigram fst for LF-MMI training
  tools/k2/prepare_mmi.sh data/train/ data/dev data/local/lfmmi

  # 9.2 Run LF-MMI training from stage 4, with below new args
  # --lfmmi_dir data/local/lfmmi

  # 9.3 Run HLG decode from stage 8.2
fi

