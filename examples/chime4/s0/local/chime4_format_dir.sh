#!/usr/bin/env bash

# wujian@2020

set -eu

echo "$0: Formating chime4 data dir..."

track=isolated_1ch_track
data_dir=data/chime4

mkdir -p $data_dir/{train,dev}

cat $data_dir/tr05_{simu,real}_noisy/wav.scp $data_dir/tr05_orig_clean/wav.scp \
  $data_dir/train_si200_wsj1_clean/wav.scp | sort -k1 > $data_dir/train/wav.scp
cat $data_dir/tr05_{simu,real}_noisy/text $data_dir/tr05_orig_clean/text \
  $data_dir/train_si200_wsj1_clean/text | sort -k1 > $data_dir/train/text

cat $data_dir/dt05_{real,simu}_${track}/wav.scp | sort -k1 > $data_dir/dev/wav.scp
cat $data_dir/dt05_{real,simu}_${track}/text | sort -k1 > $data_dir/dev/text

echo "$0: Format $data_dir done"
