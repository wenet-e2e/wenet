#!/usr/bin/env bash

set -eu

[ $# -ne 2 ] && echo "Script format error: $0 <data-dir> <dump-dir>" && exit 0

data_dir=$1
dump_dir=$2

mkdir -p $dump_dir

num_utts=$(cat $data_dir/wav.scp | wc -l)
echo "Orginal utterances (.wav + .wv1): $num_utts"

# cat $data_dir/wav.scp | grep "sph2pipe" | \
#   awk -v dir=$dump_dir '{printf("%s -f wav %s %s/%s.wav\n", $2, $5, dir, $1)}' | bash

awk '{print $1,$5}' $data_dir/wav.scp > $data_dir/raw_wav.scp
find $dump_dir -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | \
  sed 's:\.wav::' > $data_dir/wav.scp

num_utts=$(cat $data_dir/wav.scp | wc -l)
echo "Wave utterances (.wav): $num_utts"

echo "$0: Generate wav => $dump_dir done"
