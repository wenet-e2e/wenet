#!/bin/bash
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang)
# Copyright    2022  Ximalaya Speech Team (author: Xiang Lyu)

lexion_dir=$1
lm_dir=$2
tgt_dir=$3

# For k2 installation, please refer to https://github.com/k2-fsa/k2/
python -c "import k2; print(k2.__file__)"
python -c "import torch; import _k2; print(_k2.__file__)"

# Prepare necessary icefall scripts
if [ ! -d tools/k2/icefall ]; then
    git clone --depth 1 https://github.com/k2-fsa/icefall.git tools/k2/icefall
fi
pip install -r tools/k2/icefall/requirements.txt
export PYTHONPATH=`pwd`/tools/k2/icefall:`pwd`/tools/k2/icefall/egs/aishell/ASR/local:$PYTHONPATH

# 8.1 Prepare char based lang
mkdir -p $tgt_dir
python tools/k2/prepare_char.py $lexion_dir/units.txt $lm_dir/wordlist $tgt_dir
echo "Compile lexicon L.pt L_disambig.pt succeeded"

# 8.2 Prepare G
mkdir -p data/lm
python -m kaldilm \
    --read-symbol-table="$tgt_dir/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    $lm_dir/lm.arpa > data/lm/G_3_gram.fst.txt

# 8.3 Compile HLG
python tools/k2/icefall/egs/aishell/ASR/local/compile_hlg.py --lang-dir $tgt_dir
echo "Compile decoding graph HLG.pt succeeded"