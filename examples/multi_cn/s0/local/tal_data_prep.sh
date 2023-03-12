#!/bin/bash

# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
# Copyright 2021 Mobvoi Inc. All Rights Reserved. (Di Wu)
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/aisolution_data data/tal_asr"
  exit 1;
fi

tal_audio_dir=$1/wav/
tal_text=$1/transcript/transcript.txt
data=$2

train_dir=$data/local/train
dev_dir=$data/local/dev
test_dir=$data/local/test
tmp_dir=$data/local/tmp

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $tal_audio_dir ] || [ ! -f $tal_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

echo "**** Creating tal asr data folder ****"

# find wav audio file for train, dev and test resp.
find $tal_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 31747 ] && \
  echo Warning: expected 31747 data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF, "TALASR"$(NF-1)"-"$NF}' > $dir/utt_uttid
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print "TALASR"$(NF-1)"-"$NF, "TALASR"$(NF-1)}' > $dir/utt2spk
  paste -d ' ' <(awk '{print $2}' $dir/utt_uttid) $dir/wav.flist > $dir/wav.scp
  tools/filter_scp.pl -f 1 $dir/utt.list $tal_text | \
    sed 's/Ａ/A/g' | sed 's/#//g' | sed 's/=//g' | sed 's/、//g' | \
    sed 's/，//g' | sed 's/？//g' | sed 's/。//g' | sed 's/[ ][ ]*$//g'\
    > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  paste -d " " <(sort -u -k 1 $dir/utt_uttid | awk '{print $2}') \
    <(sort -u -k 1 $dir/transcripts.txt | awk '{for(i=2;i<NF;i++) {printf($i" ")}printf($NF"\n") }') \
    > $dir/text
  tools/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p $data/train $data/dev $data/test

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f $data/train/$f || exit 1;
  cp $dev_dir/$f $data/dev/$f || exit 1;
  cp $test_dir/$f $data/test/$f || exit 1;
done

tools/fix_data_dir.sh $data/train || exit 1;
tools/fix_data_dir.sh $data/dev || exit 1;
tools/fix_data_dir.sh $data/test || exit 1;

echo "$0: tal asr data preparation succeeded"
exit 0;
