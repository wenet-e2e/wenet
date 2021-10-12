export WENET_DIR=/apdcephfs/share_1157259/users/yougenyuan/backup/handover/wenet
export BUILD_DIR=${WENET_DIR}/runtime/server/x86/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${PWD}/utils:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
#export PYTHONPATH=
export PYTHONPATH=/apdcephfs/share_1157259/users/yougenyuan/software/miniconda3/lib/python3.8/site-packages:$PYTHONPATH

export KALDI_ROOT=/apdcephfs/share_1157259/users/yougenyuan/backup/kaldi-master-20021204
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh                   
export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/tools/F4DE/bin:$PATH
#[ -f $KALDI_ROOT/tools/config/common_path.sh ] && . $KALDI_ROOT/tools/config/common_path.sh
. /apdcephfs/share_1157259/users/janinezhao/kaldi/tools/config/common_path.sh

export LC_ALL=C
