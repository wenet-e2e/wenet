export WENET_DIR=$PWD/../../..
export PATH=$PWD:${WENET_DIR}/runtime/server/x86/build/:${WENET_DIR}/runtime/server/x86/build/kaldi/:${WENET_DIR}/runtime/server/x86/build/openfst/bin/:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
