#!/usr/bin/env bash
# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)
export GLOG_logtostderr=1
export GLOG_v=2

set -e

nj=1
chunk_size=-1
ctc_weight=0.0
reverse_weight=0.0
rescoring_weight=1.0
# For CTC WFST based decoding
fst_path=
acoustic_scale=1.0
beam=15.0
lattice_beam=12.0
min_active=200
max_active=7000
blank_skip_thresh=1.0
length_penalty=0.0

. tools/parse_options.sh || exit 1;
if [ $# != 5 ]; then
  echo "Usage: $0 [options] <wav.scp> <label_file> <model_file> <dict_file> <output_dir>"
  exit 1;
fi

if ! which decoder_main > /dev/null; then
  echo "decoder_main is not built, please go to runtime/server/x86 to build it."
  exit 1;
fi

scp=$1
label_file=$2
model_file=$3
dict_file=$4
dir=$5

mkdir -p $dir/split${nj}

# Step 1. Split wav.scp
split_scps=""
for n in $(seq ${nj}); do
  split_scps="${split_scps} ${dir}/split${nj}/wav.${n}.scp"
done
tools/data/split_scp.pl ${scp} ${split_scps}

# Step 2. Parallel decoding
wfst_decode_opts=
if [ ! -z $fst_path ]; then
  wfst_decode_opts="--fst_path $fst_path"
  wfst_decode_opts="$wfst_decode_opts --beam $beam"
  wfst_decode_opts="$wfst_decode_opts --lattice_beam $lattice_beam"
  wfst_decode_opts="$wfst_decode_opts --max_active $max_active"
  wfst_decode_opts="$wfst_decode_opts --min_active $min_active"
  wfst_decode_opts="$wfst_decode_opts --acoustic_scale $acoustic_scale"
  wfst_decode_opts="$wfst_decode_opts --blank_skip_thresh $blank_skip_thresh"
  wfst_decode_opts="$wfst_decode_opts --length_penalty $length_penalty"
  echo $wfst_decode_opts > $dir/config
fi
for n in $(seq ${nj}); do
{
  decoder_main \
     --rescoring_weight $rescoring_weight \
     --ctc_weight $ctc_weight \
     --reverse_weight $reverse_weight \
     --chunk_size $chunk_size \
     --wav_scp ${dir}/split${nj}/wav.${n}.scp \
     --model_path $model_file \
     --dict_path $dict_file \
     $wfst_decode_opts \
     --result ${dir}/split${nj}/${n}.text &> ${dir}/split${nj}/${n}.log
} &
done
wait

# Step 3. Merge files
for n in $(seq ${nj}); do
  cat ${dir}/split${nj}/${n}.text
done > ${dir}/text
tail $dir/split${nj}/*.log | grep RTF | awk '{sum+=$NF}END{print sum/NR}' > $dir/rtf

# Step 4. Compute WER
python3 tools/compute-wer.py --char=1 --v=1 \
  $label_file $dir/text > $dir/wer
