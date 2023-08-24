export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_BIN=${BUILD_DIR}/../fc_base/openfst-build/src
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_BIN}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH
