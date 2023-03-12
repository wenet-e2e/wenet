#!/bin/bash
# convert sph scp to segmented wav scp
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

sph2pipe_version="v2.5"
if [ ! -d tools/sph2pipe_${sph2pipe_version} ]; then
  echo "Download sph2pipe_${sph2pipe_version} ......"
  wget -T 10 -t 3 -P tools https://www.openslr.org/resources/3/sph2pipe_${sph2pipe_version}.tar.gz || \
  wget -T 10 -c -P tools https://sourceforge.net/projects/kaldi/files/sph2pipe_${sph2pipe_version}.tar.gz; \
  tar --no-same-owner -xzf tools/sph2pipe_${sph2pipe_version}.tar.gz -C tools
  cd tools/sph2pipe_${sph2pipe_version}/ && \
        gcc -o sph2pipe  *.c -lm
  cd -
fi
sph2pipe=`which sph2pipe` || sph2pipe=`pwd`/tools/sph2pipe_${sph2pipe_version}/sph2pipe
[ ! -x $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;
sox=`which sox`
[ ! -x $sox ] && echo "Could not find the sox program at $sph2pipe" && exit 1;

cat $inscp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s#-f#wav#-p#-c#1#%s#|\n", $1, sph2pipe, $2);
    printf("%s-B %s#-f#wav#-p#-c#2#%s#|\n", $1, sph2pipe, $2);}' | \
   sort > $data/wav_ori.scp || exit 1;

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
