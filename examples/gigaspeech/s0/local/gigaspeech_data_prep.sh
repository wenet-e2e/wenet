#!/usr/bin/env bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)
#                 Mobvoi Corporation (Author: Di Wu)

set -e
set -o pipefail

stage=1
prefix=
garbage_utterance_tags="<SIL> <MUSIC> <NOISE> <OTHER>"
punctuation_tags="<COMMA> <EXCLAMATIONPOINT> <PERIOD> <QUESTIONMARK>"
train_subset=XL

. ./tools/parse_options.sh || exit 1;

filter_by_id () {
  idlist=$1
  input=$2
  output=$3
  field=1
  if [ $# -eq 4 ]; then
    field=$4
  fi
  cat $input | perl -se '
    open(F, "<$idlist") || die "Could not open id-list file $idlist";
    while(<F>) {
      @A = split;
      @A>=1 || die "Invalid id-list file line $_";
      $seen{$A[0]} = 1;
    }
    while(<>) {
      @A = split;
      @A > 0 || die "Invalid file line $_";
      @A >= $field || die "Invalid file line $_";
      if ($seen{$A[$field-1]}) {
        print $_;
      }
    }' -- -idlist="$idlist" -field="$field" > $output ||\
  (echo "$0: filter_by_id() error: $input" && exit 1) || exit 1;
}

subset_data_dir () {
  utt_list=$1
  src_dir=$2
  dest_dir=$3
  mkdir -p $dest_dir || exit 1;
  # wav.scp text segments utt2dur
  filter_by_id $utt_list $src_dir/utt2dur $dest_dir/utt2dur ||\
    (echo "$0: subset_data_dir() error: $src_dir/utt2dur" && exit 1) || exit 1;
  filter_by_id $utt_list $src_dir/text $dest_dir/text ||\
    (echo "$0: subset_data_dir() error: $src_dir/text" && exit 1) || exit 1;
  filter_by_id $utt_list $src_dir/segments $dest_dir/segments ||\
    (echo "$0: subset_data_dir() error: $src_dir/segments" && exit 1) || exit 1;
  awk '{print $2}' $dest_dir/segments | sort | uniq > $dest_dir/reco
  filter_by_id $dest_dir/reco $src_dir/wav.scp $dest_dir/wav.scp ||\
    (echo "$0: subset_data_dir() error: $src_dir/wav.scp" && exit 1) || exit 1;
  rm -f $dest_dir/reco
}

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <gigaspeech-dataset-dir> <data-dir>"
  echo " e.g.: $0 --train-subset XL /disk1/audio_data/gigaspeech/ data/"
  echo ""
  echo "This script takes the GigaSpeech source directory, and prepares the"
  echo "WeNet format data directory."
  echo "  --garbage-utterance-tags <tags>  # Tags for non-speech."
  echo "  --prefix <prefix>                # Prefix for output data directory."
  echo "  --punctuation-tags <tags>        # Tags for punctuations."
  echo "  --stage <stage>                  # Processing stage."
  echo "  --train-subset <XL|L|M|S|XS>     # Train subset to be created."
  exit 1
fi

gigaspeech_dir=$1
data_dir=$2

declare -A subsets
subsets=(
  [XL]="train_xl"
  [L]="train_l"
  [M]="train_m"
  [S]="train_s"
  [XS]="train_xs"
  [DEV]="dev"
  [TEST]="test")
prefix=${prefix:+${prefix}_}

corpus_dir=$data_dir/${prefix}corpus/
if [ $stage -le 1 ]; then
  echo "$0: Extract meta into $corpus_dir"
  # Sanity check.
  [ ! -f $gigaspeech_dir/GigaSpeech.json ] &&\
    echo "$0: Please download $gigaspeech_dir/GigaSpeech.json!" && exit 1;
  [ ! -d $gigaspeech_dir/audio ] &&\
    echo "$0: Please download $gigaspeech_dir/audio!" && exit 1;

  [ ! -d $corpus_dir ] && mkdir -p $corpus_dir

  # Files to be created:
  # wav.scp text segments utt2dur
  python3 local/extract_meta.py \
     $gigaspeech_dir/GigaSpeech.json $corpus_dir || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: Filter $corpus_dir/text"
  # Delete utterances with garbage meta tags
  for tag in $garbage_utterance_tags; do
    sed -i "/${tag}/d" $corpus_dir/text
  done

  # Delete punctuations in utterances
  for tag in $punctuation_tags; do
    sed -i "s/${tag}//g" $corpus_dir/text
  done

  # Ensure space only appears once and utt is seprated with others by '\t'
  sed -i 's/\t/ /g' $corpus_dir/text
  sed -i 's/[ ][ ]*/ /g' $corpus_dir/text
  sed -i 's/ /\t/' $corpus_dir/text
fi

if [ $stage -le 3 ]; then
  echo "$0: Split data to train, dev and test"
  # Split data to train, dev and test.
  [ ! -f $corpus_dir/utt2subsets ] &&\
    echo "$0: No such file $corpus_dir/utt2subsets!" && exit 1;
  for label in $train_subset DEV TEST; do
    if [ ! ${subsets[$label]+set} ]; then
      echo "$0: Subset $label is not defined in GigaSpeech.json." && exit 1;
    fi
    subset=${subsets[$label]}
    [ ! -d $data_dir/${prefix}$subset ] && mkdir -p $data_dir/${prefix}$subset
    grep "{$label}" $corpus_dir/utt2subsets \
      > $corpus_dir/${prefix}${subset}_utt_list|| exit 1;
    subset_data_dir $corpus_dir/${prefix}${subset}_utt_list \
      $corpus_dir $data_dir/${prefix}$subset || exit 1;
  done
fi

echo "$0: Done"
