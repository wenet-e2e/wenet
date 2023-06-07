#!/bin/bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Xingyu Na
#             2023  Zhanghognbin
# Apache 2.0
'''
1、解析脚本传入的命令行参数。
2、判断指定的本地目录是否存在。
3、判断指定的部分数据集名称是否正确，当前支持 data_aishell 和 resource_aishell 两个选项。
4、判断指定的远程服务器地址是否为空。
5、判断指定的部分数据集是否已经成功解压到了本地目录中。如果是，则直接退出脚本。
6、判断指定的部分数据集是否已经下载到了本地目录中。如果已经下载，检查文件大小是否一致，如果不一致则删除该文件。
7、如果指定的部分数据集还未下载，则使用 wget 工具从远程服务器下载对应的压缩文件。
8、对下载的压缩文件进行解压操作，并在本地目录标记该部分数据集已经解压完毕。
9、如果下载的是 data_aishell 部分数据集，则在对应子目录下进行额外的处理，即解压每个 .tar.gz 格式的音频文件。
10、根据指定的命令行参数，在成功解压并处理后可选择删除下载的压缩文件。
11、执行完毕后退出脚本。
'''
remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <corpus-part>"
  echo "e.g.: $0 /export/a05/xna/data www.openslr.org/resources/33 data_aishell"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: data_aishell, resource_aishell."
fi

data=$1
url=$2
part=$3

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

part_ok=false
list="data_aishell resource_aishell"
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

# sizes of the archive files in bytes.
sizes="15582913665 1246920"

if [ -f $data/$part.tgz ]; then
  size=$(/bin/ls -l $data/$part.tgz | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tgz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$part.tgz
  else
    echo "$data/$part.tgz exists and appears to be complete."
  fi
fi

if [ ! -f $data/$part.tgz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url/$part.tgz
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

cd $data

if ! tar -xvzf $part.tgz; then
  echo "$0: error un-tarring archive $data/$part.tgz"
  exit 1;
fi

touch $data/$part/.complete

if [ $part == "data_aishell" ]; then
  cd $data/$part/wav
  for wav in ./*.tar.gz; do
    echo "Extracting wav from $wav"
    tar -zxf $wav && rm $wav
  done
fi

echo "$0: Successfully downloaded and un-tarred $data/$part.tgz"

if $remove_archive; then
  echo "$0: removing $data/$part.tgz file since --remove-archive option was supplied."
  rm $data/$part.tgz
fi

exit 0;
