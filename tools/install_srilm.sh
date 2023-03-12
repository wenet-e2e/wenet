#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#           2022  Binbin Zhang(binbzha@qq.com)

current_path=`pwd`
current_dir=`basename "$current_path"`

if [ "tools" != "$current_dir" ]; then
  echo "You should run this script in tools/ directory!!"
  exit 1
fi

! command -v gawk > /dev/null && \
   echo "GNU awk is not installed so SRILM will probably not work correctly: refusing to install" && exit 1;

srilm_url="https://github.com/BitSpeech/SRILM/archive/refs/tags/1.7.3.tar.gz"

if [ ! -f ./srilm.tar.gz ];  then
  if ! wget -O ./srilm.tar.gz "$srilm_url"; then
    echo 'There was a problem downloading the file.'
    echo 'Check you internet connection and try again.'
    exit 1
  fi
fi

tar -zxvf srilm.tar.gz
mv SRILM-1.7.3 srilm

# set the SRILM variable in the top-level Makefile to this directory.
cd srilm
cp Makefile tmpf

cat tmpf | gawk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' \
  > Makefile || exit 1
rm tmpf

make || exit
cd ..

(
  [ ! -z "${SRILM}" ] && \
    echo >&2 "SRILM variable is aleady defined. Undefining..." && \
    unset SRILM

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${SRILM}" ] && \
    echo >&2 "SRILM config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export SRILM=$wd/srilm"
  dirs="\${PATH}"
  for directory in $(cd srilm && find bin -type d ) ; do
    dirs="$dirs:\${SRILM}/$directory"
  done
  echo "export PATH=$dirs"
) >> env.sh

echo >&2 "Installation of SRILM finished successfully"
echo >&2 "Please source the tools/env.sh in your path.sh to enable it"
