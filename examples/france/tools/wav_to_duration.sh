#!/bin/bash
# split the wav scp, calculate duration and merge
nj=4
. tools/parse_options.sh || exit 1;

inscp=$1
outscp=$2
data=$(dirname ${inscp})
if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/log
fi
mkdir -p ${logdir}

rm -f $logdir/wav_*.slice
rm -f $logdir/wav_*.shape
split --additional-suffix .slice -d -n l/$nj $inscp $logdir/wav_

for slice in `ls $logdir/wav_*.slice`; do
{
    name=`basename -s .slice $slice`
    tools/wav2dur.py $slice $logdir/$name.shape 1>$logdir/$name.log
} &
done
wait
cat $logdir/wav_*.shape > $outscp
