#!/usr/bin/bash
stage=-1
stop_stage=1
wenet_dir=./wenet

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    git clone https://github.com/wenet-e2e/wenet.git $wenet_dir
    src=$wenet_dir/runtime/libtorch
    mkdir -p $src/build
    cmake -B $src/build -S $src -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON -DONNX=ON -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF && cmake --build $src/build
fi

export WENET_DIR=$wenet_dir
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_BIN=${BUILD_DIR}/../fc_base/openfst-build/src
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_BIN}/bin:$PATH

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare dict
  git lfs install
  git clone https://huggingface.co/yuekai/aishell1_tlg_essentials
  mkdir -p data/local/dict data/local/lm data/local/lang
  unit_file=./aishell1_tlg_essentials/units.txt
  cp $unit_file data/local/dict/units.txt
  ${wenet_dir}/tools/fst/prepare_dict.py $unit_file ./aishell1_tlg_essentials/resource_aishell/lexicon.txt \
    data/local/dict/lexicon.txt
  # using pretrained lm
  cp ./aishell1_tlg_essentials/3-gram.unpruned.arpa data/local/lm/lm.arpa
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Build decoding TLG
  ln -s ${wenet_dir}/tools ./
  ln -s /usr/bin/python3 /usr/bin/python
  tools/fst/compile_lexicon_token_fst.sh \
    data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
fi

