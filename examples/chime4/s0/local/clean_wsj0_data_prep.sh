#!/usr/bin/env bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# Modified from Kaldi's chime4 recipe

set -eu

dataset=chime4

. ./tools/parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <original WSJ0 corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level WSJ corpus directory."
  echo "It is assumed that there will be a 'wsj0' and a 'wsj1' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

wsj0=$1

srcdir=$PWD/data/chime4/local
dstdir=$PWD/data/$dataset
local=$PWD/local
utils=$PWD/utils
sph2pipe=sph2pipe

if [ ! `which sph2pipe` ]; then
  echo "Could not find sph2pipe, install it first..."
  mkdir -p exp && cd exp && wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
  tar -zxf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm && cd .. && rm -rf sph2pipe_v2.5.tar.gz
  sph2pipe=$PWD/sph2pipe_v2.5/sph2pipe
  cd ..
fi

mkdir -p $srcdir && cd $srcdir

# This version for SI-84
cat $wsj0/wsj0/doc/indices/train/tr_s_wv1.ndx \
  | $local/cstr_ndx2flist.pl $wsj0 | sort -u > tr05.flist

# Now for the test sets.
# $wsj0/wsj1/doc/indices/readme.doc
# describes all the different test sets.
# Note: each test-set seems to come in multiple versions depending
# on different vocabulary sizes, verbalized vs. non-verbalized
# pronunciations, etc.  We use the largest vocab and non-verbalized
# pronunciations.
# The most normal one seems to be the "baseline 60k test set", which
# is h1_p0.

# Nov'92 (330 utts, 5k vocab)
cat $wsj0/wsj0/doc/indices/test/nvp/si_et_05.ndx | \
  $local/cstr_ndx2flist.pl $wsj0 | sort > et05.flist

# Note: the ???'s below match WSJ and SI_DT, or wsj and si_dt.
# Sometimes this gets copied from the CD's with upcasing, don't know
# why (could be older versions of the disks).
find $wsj0/wsj0/si_dt_05 -print | grep -i ".wv1" | sort > dt05.flist

# Finding the transcript files:
find -L $wsj0 -iname '*.dot' > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
# adding suffix to utt_id
# 0 for clean condition
for x in tr05 et05 dt05; do
  $local/flist2scp.pl $x.flist | sort > ${x}_sph_tmp.scp
  cat ${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > ${x}_tmp.trans1
  cat ${x}_sph_tmp.scp | awk '{printf("%s %s\n", $1, $2);}' > ${x}_sph.scp
  cat ${x}_tmp.trans1 | awk '{printf("%s ", $1); for(i=2;i<=NF;i++) printf("%s ", $i); printf("\n");}' > ${x}.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in tr05 et05 dt05; do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in tr05 et05 dt05; do
  awk -v cmd=$sph2pipe '{printf("%s %s -f wav %s |\n", $1, cmd, $2);}' ${x}_sph.scp > ${x}_wav.scp
done

if [ ! -f wsj0-train-spkrinfo.txt ] || [ `cat wsj0-train-spkrinfo.txt | wc -l` -ne 134 ]; then
  rm -f wsj0-train-spkrinfo.txt
  wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt \
    || ( echo "Getting wsj0-train-spkrinfo.txt from backup location" && \
         wget --no-check-certificate https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt );
fi

if [ ! -f wsj0-train-spkrinfo.txt ]; then
  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
  echo "This is possibly omitted from the training disks; couldn't find it."
  echo "Everything else may have worked; we just may be missing gender info"
  echo "which is only needed for VTLN-related diagnostics anyway."
  exit 1
fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.

cat $wsj0/wsj0/doc/spkrinfo.txt \
    ./wsj0-train-spkrinfo.txt  | \
    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
    awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender

# return back
cd -

for x in et05 dt05 tr05; do
  mkdir -p $dstdir/${x}_orig_clean
  cp $srcdir/$x.txt $dstdir/${x}_orig_clean/text || exit 1
  cp $srcdir/${x}_wav.scp $dstdir/${x}_orig_clean/wav.scp || exit 1
done

echo "Data preparation succeeded"
