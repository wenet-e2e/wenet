#!/bin/bash

# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
# Copyright 2021 Mobvoi Inc. All Rights Reserved. (Di Wu)
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/TAL_CSASR data/tal_mix"
  exit 1;
fi

tal_mix_audio_dir=$1/cs_wav
tal_mix_text=$1/label
data=$2

train_dir=$data/local/train
tmp_dir=$data/local/tmp

mkdir -p $train_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $tal_mix_audio_dir ] || [ ! -f $tal_mix_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

echo "**** Creating tal mix data folder ****"

# find wav audio file for train, dev and test resp.
find $tal_mix_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 370000 ] && \
  echo Warning: expected 370000 data files, found $n

# rm -r $tmp_dir

# Transcriptions preparation
echo Preparing transcriptions
sed -e 's/\.wav//' $tmp_dir/wav.flist | awk -F '/' '{print $NF}' > $train_dir/utt.list
sed -e 's/\.wav//' $tmp_dir/wav.flist | awk -F '/' '{printf("%s %s\n",$NF,$NF)}' > $train_dir/utt2spk
paste -d' ' $train_dir/utt.list $tmp_dir/wav.flist > $train_dir/wav.scp
cat $tal_mix_text  | grep -Ev '^\s*$' | awk '{if(NF>1) print $0}' > $train_dir/transcript.txt
#cp $tal_mix_text $train_dir

wc -l $train_dir/transcript.txt
echo filtering
tools/filter_scp.pl -f 1 $train_dir/utt.list $train_dir/transcript.txt | \
  sed 's/Ａ/A/g' | sed 's/Ｃ/C/g' | sed 's/Ｄ/D/g' | sed 's/Ｇ/G/g' | \
  sed 's/Ｈ/H/g' | sed 's/Ｕ/U/g' | sed 's/Ｙ/Y/g' | sed 's/ａ/a/g' | \
  sed 's/Ｉ/I/g' | sed 's/#//g' | sed 's/=//g' | sed 's/；//g' | \
  sed 's/，//g' | sed 's/？//g' | sed 's/。//g' | sed 's/\///g' | \
  sed 's/！//g' | sed 's/!//g' | sed 's/\.//g' | sed 's/\?//g' | \
  sed 's/：//g' | sed 's/,//g' | sed 's/\"//g' | sed 's/://g' | \
  sed 's/@//g' | sed 's/-/ /g' | sed 's/、/ /g' | sed 's/~/ /g' | \
  sed "s/‘/\'/g" | sed 's/Ｅ/E/g' | sed "s/’/\'/g" | sed 's/《//g' | sed 's/》//g' | \
  sed "s/[ ][ ]*$//g" | sed "s/\[//g" | sed 's/、//g' > $train_dir/text
tools/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt

mkdir -p $data/train

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f $data/train/$f || exit 1;
done

tools/fix_data_dir.sh $data/train || exit 1;

echo "$0: tal mix data preparation succeeded"
exit 0;
