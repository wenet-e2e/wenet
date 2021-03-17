#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# transform raw AISHELL-2 data to kaldi format

if [ $# != 3 ]; then
  echo "prepare_data.sh <corpus-data-dir> <tmp-dir> <output-dir>"
  echo " e.g prepare_data.sh /data/AISHELL-2/iOS/train data/local/train data/train"
  exit 1;
fi

corpus=$1
tmp=$2
dir=$3

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $tmp
mkdir -p $dir

# corpus check
if [ ! -d $corpus ] || [ ! -f $corpus/wav.scp ] || [ ! -f $corpus/trans.txt ]; then
  echo "Error: $0 requires wav.scp and trans.txt under $corpus directory."
  exit 1;
fi

# validate utt-key list
awk '{print $1}' $corpus/wav.scp   > $tmp/wav_utt.list
awk '{print $1}' $corpus/trans.txt > $tmp/trans_utt.list
tools/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/trans_utt.list > $tmp/utt.list

# wav.scp
awk -F'\t' -v path_prefix=$corpus '{printf("%s\t%s/%s\n",$1,path_prefix,$2)}' $corpus/wav.scp > $tmp/tmp_wav.scp
tools/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

# text
tools/filter_scp.pl -f 1 $tmp/utt.list $corpus/trans.txt | sort -k 1 | uniq > $tmp/trans.txt
dos2unix < $tmp/trans.txt | \
  tools/filter_scp.pl -f 1 $tmp/utt.list - | \
  sort -k 1 | uniq | tr '[a-z]' '[A-Z]' | \
  sed 's/Ａ/A/g' | sed 's/Ｔ/T/g' | sed 's/Ｍ/M/g' | sed 's/𫚉//g' | sed 's/𫖯/頫/g' | \
  sed 's/[()]//g' | sed "s/\([^A-Z]\)'/\1/g" > $tmp/text

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text; do
  cp $tmp/$f $dir/$f || exit 1;
done

echo "local/prepare_data.sh succeeded"
exit 0;

