#!/bin/bash
# dumps such pipe-style-wav to real audio file
nj=1
. tools/parse_options.sh || exit 1;

inscp=$1
segments=$2
outscp=$3
data=$(dirname ${inscp})
if [ $# -eq 4 ]; then
  logdir=$4
else
  logdir=${data}/log
fi
mkdir -p ${logdir}

sox=`which sox`
[ ! -x $sox ] && echo "Could not find the sox program at $sph2pipe" && exit 1;

paste -d " " <(cut -f 1 -d " " $inscp) <(cut -f 2- -d " " $inscp | tr -t " " "#") \
    > $data/wav_ori.scp

tools/segment.py --segments $segments --input $data/wav_ori.scp --output $data/wav_segments.scp
sed -i 's/ /,/g' $data/wav_segments.scp
sed -i 's/#/ /g' $data/wav_segments.scp

rm -f $logdir/wav_*.slice
rm -f $logdir/*.log
split --additional-suffix .slice -d -n l/$nj $data/wav_segments.scp $logdir/wav_

for slice in `ls $logdir/wav_*.slice`; do
{
    name=`basename -s .slice $slice`
    mkdir -p ${data}/wavs/${name}
    cat ${slice} | awk -F ',' -v sox=$sox -v data=`pwd`/$data/wavs/$name \
        -v logdir=$logdir -v name=$name '{
        during=$4-$3
        cmd=$2 sox " - " data "/" $1 ".wav" " trim " $3 " " during;
        system(cmd)
        printf("%s %s/%s.wav\n", $1, data, $1);
        }' | \
       sort > ${data}/wavs_${name}.scp || exit 1;
} &
done
wait
cat ${data}/wavs_*.scp > $outscp
rm ${data}/wavs_*.scp

rm -f $data/{segments,wav_segments.scp,reco2file_and_channel,reco2dur}
tools/fix_data_dir.sh $data
