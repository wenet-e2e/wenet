#! /bin/bash
# Do the alignment with CTC to get the time step

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=16
cmd=run.pl
batch=0

.tools/parse_options.sh
if [ $# -ne 3 ]; then
    echo "Usage: $0 <data-dir> <model> <align-dir>"
    exit 1;
fi

datadir=$1
model=$2
output=$3

modeldir=$(dirname $model)

mkdir -p $output
for f in $datadir/format.data $model $modeldir/train.yaml $
if [ ! -f $datadir/format.data ]; then
    echo "Error: $f doesn't exist!"
    exit 1;
fi

opts=
if [ $batch -ne 0 ]; then
    opts+="--batch_size $batch"