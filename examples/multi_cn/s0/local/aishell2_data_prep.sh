#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# This script is copied from aishell2/s5/local/prepare_data.sh
# but using difference word segmentation script.

# transform raw AISHELL-2 data to kaldi format

. ./path.sh || exit 1;

tmp=
dir=

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-data-dir> <output-dir>"
  echo " $0 /export/AISHELL-2/iOS/train data/train"
  exit 1;
fi

corpus=$1
dir=$2
tmp=$dir/tmp

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $dir
mkdir -p $tmp


# corpus check
if [ ! -d $corpus ] || [ ! -f $corpus/wav.scp ] || [ ! -f $corpus/trans.txt ]; then
  echo "Error: $0 requires wav.scp and trans.txt under $corpus directory."
  exit 1;
fi

# validate utt-key list
awk '{print "AISHELL2_"$1}' $corpus/wav.scp   > $tmp/wav_utt.list
awk '{print "AISHELL2_"$1}' $corpus/trans.txt > $tmp/trans_utt.list
tools/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/trans_utt.list > $tmp/utt.list

# wav.scp
awk -F'\t' -v path_prefix=$corpus '{printf("AISHELL2_%s %s/%s\n",$1,path_prefix,$2)}' $corpus/wav.scp > $tmp/tmp_wav.scp
tools/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

awk -F'\t' '{printf("AISHELL2_%s %s\n",$1,$2)}' $corpus/trans.txt > $tmp/tmp_trans.txt
tools/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_trans.txt | sort -k 1 | uniq > $tmp/trans.txt

# text has ' sed "s/'//g"
dos2unix < $tmp/trans.txt | \
  tools/filter_scp.pl -f 1 $tmp/utt.list - | \
  sort -k 1 | uniq | tr '[a-z]' '[A-Z]' | \
  sed 's/Ａ/A/g' | sed 's/Ｔ/T/g' | sed 's/Ｍ/M/g' | sed 's/𫚉//g' | sed 's/𫖯/頫/g' \
  > $tmp/text

# utt2spk & spk2utt
awk -F' ' '{print $2}' $tmp/wav.scp > $tmp/wav.list
sed -e 's:\.wav::g' $tmp/wav.list | \
  awk -F'/' '{i=NF-1;printf("AISHELL2_%s AISHELL2_%s\n",$NF,$i)}' > $tmp/tmp_utt2spk
tools/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
tools/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text spk2utt utt2spk; do
  cp $tmp/$f $dir/$f || exit 1;
done

tools/validate_data_dir.sh --no-feats $dir || exit 1;
echo "local/prepare_data.sh succeeded"
exit 0;
