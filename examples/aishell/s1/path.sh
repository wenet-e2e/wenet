#ANACONDA_ROOT=/home/binbinzhang/miniconda3

export PATH=$PWD/utils/:$PWD:$PATH

export LC_ALL=C

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
#source $ANACONDA_ROOT/bin/activate py3
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=~/e2e/wenet/:$PYTHONPATH
