#!/usr/bin/env bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

set -eu


if [ $# -ne 1 ]; then
  echo "Arguments should be WSJ1 directory"
  exit 1;
fi

wsj1=$1
dir=$PWD/data/chime4/local
odir=$PWD/data/chime4
mkdir -p $dir
local=$PWD/local
sph2pipe=sph2pipe

if [ ! `which sph2pipe` ]; then
  echo "Could not find sph2pipe, install it first..."
  mkdir -p exp && cd exp && wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
  tar -zxf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm && cd .. && rm -rf sph2pipe_v2.5.tar.gz
  sph2pipe=$PWD/sph2pipe_v2.5/sph2pipe
  cd ..
fi

cd $dir
# This version for SI-200
cat $wsj1/13-34.1/wsj1/doc/indices/si_tr_s.ndx | \
 $local/ndx2flist.pl $wsj1/??-{?,??}.? | sort > train_si200.flist

nl=`cat train_si200.flist | wc -l`
[ "$nl" -eq 30278 ] || echo "Warning: expected 30278 lines in train_si200.flist, got $nl"

# Dev-set for Nov'93 (503 utts)
cat $wsj1/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  $local/ndx2flist.pl $wsj1/??-{?,??}.? | sort > test_dev93.flist

# Finding the transcript files:
for x in $wsj1/??-{?,??}.?; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for x in train_si200 test_dev93; do
   $local/flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' | $local/find_transcripts.pl  dot_files.flist > $x.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si200 test_dev93; do
   cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si200 test_dev93; do
  awk -v cmd=$sph2pipe '{printf("%s %s -f wav %s |\n", $1, cmd, $2);}' ${x}_sph.scp > ${x}_wav.scp
done

# return back
cd -

for x in train_si200 test_dev93; do
  mkdir -p $odir/${x}_wsj1_clean
  cp $dir/$x.txt $odir/${x}_wsj1_clean/text || exit 1
  cp $dir/${x}_wav.scp $odir/${x}_wsj1_clean/wav.scp || exit 1
done

echo "Data preparation WSJ1 succeeded"
