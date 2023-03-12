#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-base> <url-base> <corpus-part>"
  echo "e.g.: $0 /home/data/aishell4 https://www.openslr.org/resources/111 train_L"
  echo "<corpus-part> can be one of: train_L, train_M, train_S, test."
fi

data=$1
url=$2
part=$3

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

part_ok=false
list="train_L train_M train_S test"
for x in $list; do
  if [ "$part" == $x ]; then part_ok=true; fi
done
if ! $part_ok; then
  echo "$0: expected <corpus-part> to be one of $list, but got '$part'"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/$part/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi

if [ -f $data/$part.tar.gz ]; then
  echo "$0: removing existing file $data/$part.tar.gz"
  rm $data/$part.tar.gz
fi

if [ ! -f $data/$part.tar.gz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/$part.tar.gz
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

cd $data

if ! tar -xvzf $part.tar.gz; then
  echo "$0: error un-tarring archive $data/$part.tgz"
  exit 1;
fi

touch $data/$part/.complete

echo "$0: Successfully downloaded and un-tarred $data/$part.tgz"

