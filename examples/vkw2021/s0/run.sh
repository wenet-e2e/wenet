#!/bin/bash
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
stage=-1
stop_stage=0

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# data
data=data
dict=data/dict/lang_char.txt
data_type=raw # raw or shard

train_set=train
dev_set=combine_dev
# Optional train_config
name=vkw_bidirect_12conformer_hs2048_output256_att4_conv2d_char
train_config=conf/train_${name}.yaml
cmvn=true
dir=exp/train_${name}_new
checkpoint= #$dir/0.pt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Data preparation
  local/vkw_data_prep.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  x=dev_5h
  for z in lgv liv stv; do
    [ ! -f data/vkw/label/lab_${z}/${x}/wav_ori.scp ] && \
      mv data/vkw/label/lab_${z}/${x}/wav.scp \
        data/vkw/label/lab_${z}/${x}/wav_ori.scp && \
      cut -d " " -f 1,4 data/vkw/label/lab_${z}/${x}/wav_ori.scp \
        > data/vkw/label/lab_${z}/${x}/wav.scp
  done
  y=`echo $x | cut -d "_" -f 1`
  mkdir -p combine_${y}
  for f in text wav.scp segments; do
    for z in lgv liv stv; do
      cat data/vkw/label/lab_${z}/${x}/$f
    done > combine_${y}/$f
  done
  # remove the space between the text labels for Mandarin dataset
  # download and transfer to wav.scp
  for x in ${dev_set} ${train_set}; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " \
      data/${x}/text.org | tr -d " ") > data/${x}/text
    rm data/${x}/text.org
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: generate segmented wav.scp and compute cmvn"
  ## For wav feature, just copy the data. Fbank extraction is done in training
  for x in ${dev_set} ${train_set}; do
    [ ! -f $data/$x/segmentd_wav.scp ] && \
      python tools/segment.py --segments $data/$x/segments \
        --input $data/$x/wav.scp \
        --output $data/$x/segmented_wav.scp
  done

  ### generate global_cmvn using training set
  tools/compute_cmvn_stats.py --num_workers 12 --train_config $train_config \
    --in_scp $data/${train_set}/segmented_wav.scp \
    --out_cmvn $data/$train_set/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Make train dict
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  tools/text2token.py -s 1 -n 1 $data/${train_set}/text | cut -f 2- -d" " | \
    tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | grep -P '[\p{Han}]'\
    | awk '{print $0 " " NR+1}' >> ${dict}

  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare requried format"
  for x in ${dev_set} ${train_set}; do
    tools/make_raw_list.py --segments $data/$x/segments \
      $data/$x/wav.scp $data/$x/text $data/$x/data.list
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
  $cmvn && cp ${data}/${train_set}/global_cmvn $dir
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
      --train_data $data/$train_set/data.list \
      --cv_data $data/${dev_set}/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 4 \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
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
  sets=${dev_set}
  keywords_list=$data/vkw/keyword/kwlist
  input_data=$feat_dir/${sets}/data.list
  checkpoint=$dir/avg_${average_num}.pt
  keyword_results=$dir/keyword_results_${sets}
  ctc_results=$dir/ctc_results_${sets}
  python3 local/vkw_kws_results.py --gpu 0 \
    --config $dir/train.yaml \
    --data_type $data_type \
    --symbol_table $dict \
    --num_workers 4 \
    --prefetch 32 \
    --input_data $input_data \
    --checkpoint $checkpoint \
    --keyword_unit_dict $keywords_list \
    --keyword_results $keyword_results \
    --ctc_results $ctc_results

  [ ! -f scripts/bin/results_to_score.sh ] && \
    ln -sf data/vkw/scripts scripts && chmod -R 755 scripts
  ### attention: install the F4DE tool before testing
  for y in "stv" "lgv" "liv"; do
    mkdir -p $dir/dev_${y}
    #[ ! -f data/vkw/score/dev_${y}/utter_map ] && \
    if [ $y == "lgv" ]; then
      grep "TV1" $keyword_results > $dir/dev_${y}/kws_results
    elif [ $y == "liv" ]; then
      grep "sph_live" $keyword_results > $dir/dev_${y}/kws_results
    elif [ $y == "stv" ]; then
      grep "sph_video" $keyword_results > $dir/dev_${y}/kws_results
    else
      "invalid $y"
    fi
    ./data/vkw/scripts/bin/results_to_score.sh \
      data/vkw/score/dev_${y}/ecf \
      data/vkw/label/lab_${y}/dev_5h/segments \
      data/vkw/score/dev_${y}/utter_map \
      $dir/dev_${y}/kws_results \
      data/vkw/keyword/kwlist.xml \
      data/vkw/score/dev_${y}/rttm
    ./data/vkw/scripts/bin/F1.sh \
      $dir/dev_${y}/kws_outputs/f4de_scores_unnormalized/alignment.csv
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "adding 5h finetune data for each scenario to obtain better results"
    local/run_finetune_5h.sh
fi
