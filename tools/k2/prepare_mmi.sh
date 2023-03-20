#!/bin/bash
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang)
# Copyright    2023  Ximalaya Speech Team (author: Xiang Lyu)

train_dir=$1
dev_dir=$2
tgt_dir=$3

# k2 and icefall updates very fast. Below commits are veryfied in this script.
# k2 3dc222f981b9fdbc8061b3782c3b385514a2d444, icefall 499ac24ecba64f687ff244c7d66baa5c222ecf0f

# For k2 installation, please refer to https://github.com/k2-fsa/k2/
python -c "import k2; print(k2.__file__)"
python -c "import torch; import _k2; print(_k2.__file__)"

# Prepare necessary icefall scripts
if [ ! -d tools/k2/icefall ]; then
    git clone --depth 1 https://github.com/k2-fsa/icefall.git tools/k2/icefall
fi
pip install -r tools/k2/icefall/requirements.txt
export PYTHONPATH=`pwd`/tools/k2/icefall:`pwd`/tools/k2/icefall/egs/aishell/ASR/local:$PYTHONPATH

# 1. prepare wordlist
mkdir -p $tgt_dir
awk 'FNR>2&&FNR<=4232{print $1}END{printf("<UNK>")}' $train_dir/units.txt > $tgt_dir/wordlist

# 2. prepare L.pt tokens.txt words.txt lexicon.txt uniq_lexicon.txt
python tools/k2/prepare_char.py $train_dir/units.txt $tgt_dir/wordlist $tgt_dir
ln -s lexicon.txt $tgt_dir/uniq_lexicon.txt

# 3. prepare token level bigram
cat $train_dir/text | awk '{print $2}'| sed -r 's/(.)/ \1/g' > $tgt_dir/transcript_chars.txt
cat $dev_dir/text | awk '{print $2}'| sed -r 's/(.)/ \1/g' >> $tgt_dir/transcript_chars.txt

./shared/make_kn_lm.py \
    -ngram-order 2 \
    -text $tgt_dir/transcript_chars.txt \
    -lm $tgt_dir/P.arpa
python -m kaldilm \
    --read-symbol-table="$tgt_dir/words.txt" \
    --disambig-symbol='#0' \
    --max-order=2 \
    $tgt_dir/P.arpa > $tgt_dir/P.fst.txt