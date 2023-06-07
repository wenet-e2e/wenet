#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0
# 过滤出在aishell_text中有转录结果的wav，取出这部分对应的转录结果，对过滤后的wav和transcript排序去重，记录到训练验证测试文件夹中的wav.scp和text

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  echo " $0 /xxx/data/data_aishell/wav /xxx/data/data_aishell/transcript"
  exit 1;
fi

aishell_audio_dir=$1
aishell_text=$2/aishell_transcript_v0.8.txt

train_dir=data/local/train  #run.sh目录下的路径
dev_dir=data/local/dev
test_dir=data/local/test
tmp_dir=data/local/tmp

mkdir -p $train_dir #-p选项指定要创建目录的路径，如果路径中的某些目录不存在，则会自动创建这些目录。如果路径中的目录已经存在，则不会报错，也不会执行任何操作
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# data directory check
# -d和-f分别表示检查目录和文件是否存在
if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: $aishell_audio_dir or $aishell_text does not exist!"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist 
# -iname：指定查找的文件名匹配模式，不区分大小写。-name选项表示区分大小写。
# "*.wav"：指定匹配的文件名模式，*表示匹配任意字符（包括空字符）
# 结果是wav文件的绝对路径

n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && echo "Warning: expected 4434 data in data_Kham, found $n"

# 使用grep命令在$tmp_dir/wav.flist文件中查找包含字符串"wav/train"的行，并将结果保存到$train_dir/wav.flist文件中
grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  # -e 表示执行替换操作。 \. 匹配字符 "."，因为 "." 在正则表达式中有特殊含义，所以需要进行转义；
  # awk 是 Linux 中的一个文本处理工具，用于对文本进行各种操作和转换。
  # -F '/' 表示使用 "/" 作为分隔符。
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list 
  # $dir/utt.list内容：音频文件id（文件名无后缀）
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all 
  # wav.scp_all内容：音频文件id 文件路径
  
  # tools/filter_scp.pl：过滤aishell_transcript_v0.8.txt文件，输出第n个字段在utt.list中的行。
  # 测试命令：
    # 标准输入中，过滤第二个字段（-f 2），如果第二个字段在（echo 1;echo 2）中，则输出整行。
    # perl filter_scp.pl -f 2 <(echo 1;echo 2) <(echo aaa 1;echo bbb 2;echo ccc 3;echo ddd 4;)
    # 结果：
    # aaa 1
    # bbb 2
  tools/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt 
  # transcripts.txt内容：音频文件id 转录结果
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list # 将没有转录结果的音频文件过滤掉
  tools/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp # sort -u（unique）将有效的filename filepath去重后记录到wav.scp
  sort -u $dir/transcripts.txt > $dir/text # 排序去重后的转录文本记录到text
done

mkdir -p data/train data/dev data/test

for f in wav.scp text; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $dev_dir/$f data/dev/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
done

echo "$0: Tibetan data preparation succeeded"
exit 0;
