#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 1. xml split by sentences
# 2. wav split by xml.simp's guidance
# 3. generate "text" and "wav.scp" files as required by wenet
# 4. compute cmvn, better wav.len >= 0.1s, otherwise bug happens...
# 5. sentence piece's bpe vocabulary
# 6. make "data.list" files
# 7. train -> 50 epochs

stage=1 # train -> 50 epochs
stop_stage=8 #

# data
#data_url=www.openslr.org/resources/12
# TODO use your own data path
datadir=/workspace/asr/csj

# output wav data dir
wave_data=data # wave file path
# Optional train_config
train_config=conf/train_conformer.yaml
checkpoint=
cmvn=true # cmvn is for mean, variance, frame_number statistics
do_delta=false # not used...

dir=exp/sp_spec_aug # model's dir (output dir)

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=10
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=4096 # TODO -> you can change this value to 5000, 100000 and so on
bpemode=bpe #unigram # TODO -> you can use unigram and other methods

set -e # if any line's exex result is not true, bash stops
set -u # show the error line when stops (failed)
set -o pipefail # return value of the whole bash = final line executed's result

train_set=train
dev_set=dev
recog_set="test1 test2 test3"

### CSJ data is not free!
# buying URL: https://ccd.ninjal.ac.jp/csj/en/

### data preparing - split xml by sentences ###
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### I did not check espnet nor kaldi for the pre-processing,
  ### I developed my own ways. so, use at your own risks.
  echo "stage 1: Data preparation -> xml preprocessing "
  echo "  -> extract [start.time, end.time, text] from raw xml files"
  python ./csj_tools/wn.0.parse.py $datadir ${wave_data}
fi

in_wav_path=$datadir/WAV
xml_simp_path=${wave_data}/xml
#wav_split_path=${wave_data}/wav.2
wav_split_path=${wave_data}/wav
mkdir -p ${wav_split_path}

### data preparing - split wav by xml.simp's guidance ###
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Data preparation -> wav preprocessing "
  echo "  -> split wav file by xml.simp's [start.time, end.time, text] format"
  # in addition, 2ch to 1ch!

  python ./csj_tools/wn.1.split_wav.py ${in_wav_path} ${xml_simp_path} ${wav_split_path}
fi

### data preparing - generate "text" and "wav.scp" files ###
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: prepare text and wav.scp for train/test1/test2/test3 from wav and xml folders"

  t1fn='list_files/test.set.1.list'
  t2fn='list_files/test.set.2.list'
  t3fn='list_files/test.set.3.list'

  outtrain=${wave_data}/train
  outt1=${wave_data}/test1
  outt2=${wave_data}/test2
  outt3=${wave_data}/test3

  mkdir -p $outtrain
  mkdir -p $outt1
  mkdir -p $outt2
  mkdir -p $outt3

  python ./csj_tools/wn.2.prep.text.py \
    ${xml_simp_path} ${wav_split_path} \
    $t1fn $t2fn $t3fn \
    $outtrain $outt1 $outt2 $outt3
fi

minsec=0.1

### compute static info: mean, variance, frame_num ###
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Feature Generation"
  # TODO if failed, then please make sure your wav files are all >= 0.1s ...

  mkdir -p $wave_data/dev
  # merge total dev data
  for set in test1 test2 test3; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/$dev_set/$f
    done
  done

  python ./csj_tools/wn.3.mincut.py $wave_data/$train_set/wav.scp $minsec

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp_$minsec \
    --out_cmvn $wave_data/$train_set/global_cmvn
fi

### use sentence piece to construct subword vocabulary ###
dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 5: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  tools/spm_train \
    --input=$wave_data/lang_char/input.txt \
    --vocab_size=${nbpe} \
    --model_type=${bpemode} \
    --model_prefix=${bpemodel} \
    --input_sentence_size=100000000

  tools/spm_encode \
    --model=${bpemodel}.model \
    --output_format=piece < $wave_data/lang_char/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Prepare wenet requried data
  echo "Prepare data, prepare requried format"
  for x in $train_set ; do
    python csj_tools/wn.4.make_raw_list.py $wave_data/$x/wav.scp_$minsec $wave_data/$x/text \
        $wave_data/$x/data.list
  done
  for x in $dev_set ${recog_set} ; do
    python csj_tools/wn.4.make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text \
        $wave_data/$x/data.list
  done
fi

### Training! ###

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn $wave_data/${train_set}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type raw \
      --symbol_table $dict \
      --train_data $wave_data/$train_set/data.list \
      --cv_data $wave_data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $num_gpus \
      --ddp.rank $i \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi

### test model ###

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
  mkdir -p $dir/test
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
  decoding_chunk_size=-1
  ctc_weight=0.5
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/${test}_${mode}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu $gpu_id \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type raw \
          --test_data $wave_data/$test/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 10 \
          --batch_size 1 \
          --penalty 0.0 \
          --dict $dict \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

        cut -f2- -d " " $test_dir/text_bpe > $test_dir/text_bpe_value_tmp
        cut -f1 -d " " $test_dir/text_bpe > $test_dir/text_bpe_key_tmp
        tools/spm_decode --model=${bpemodel}.model --input_format=piece \
          < $test_dir/text_bpe_value_tmp | sed -e "s/▁/ /g" > $test_dir/text_value_tmp
        paste -d " " $test_dir/text_bpe_key_tmp $test_dir/text_value_tmp > $test_dir/text

        python tools/compute-wer.py --char=1 --v=1 \
          $wave_data/$test/text $test_dir/text > $test_dir/wer
      } &

      ((idx+=1))
      if [ $idx -eq $num_gpus ]; then
        idx=0
      fi
    }
    done
  done
  wait
fi

