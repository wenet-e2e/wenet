#!/usr/bin/env bash
if [ $# -le 0 ]; then
    echo "Argument should be France src directory, see ../run.sh for example."
    exit 1;
fi
dir=`pwd`/data
local=`pwd`/local
src_path=$1
if [ ! -d ${dir} ]; then
    mkdir ${dir}
  else
    rm -rf ${dir}
    mkdir ${dir}
fi

for x in train dev test; do
    if [ ! ${dir}/${x} ]; then
        mkdir ${dir}/${x}
    else
        rm -rf ${dir}/${x}
        mkdir ${dir}/${x}
    fi
done

if [ ! -d ${src_path}/wavs ]; then
    mkdir ${src_path}/wavs
fi
for x in train dev test; do
    python3 ${local}/create_scp_text.py  ${src_path} ${x} ${dir}/${x}
done
