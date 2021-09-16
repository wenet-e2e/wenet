
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

stage=0
stop_stage=1

# test_feat_dir will be generated in preprocessing step
# please refer to run.sh
test_feat_dir=raw_wav/test

# dir is model directory, which can be
# downloaded from https://github.com/wenet-e2e/wenet/tree/main/examples/aishell2/s0
# if you don't have one
# Please edit ctc weight, reverse weight, global_cmvn path in config.yaml or train.yaml

dir=20210618_u2pp_conformer_exp
beam_size=10
decode_modes="ctc_prefix_beam_search attention_rescoring"
dict=$dir/words.txt

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        python3 -c 'import onnxruntime' || exit 1;
        echo "export to offline onnx"
        python3 wenet/bin/export_onnx.py --config $dir/train.yaml \
            --checkpoint $dir/final.pt \
            --beam_size $beam_size \
            --output_onnx_directory $dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # check if ctc prefix beam search decoder is installed
    python3 -c 'import swig_decoders' || exit 1;

    echo "infer offline onnx"
    for mode in ${decode_modes}; do
    {
        test_dir=$dir/test_${mode}
        mkdir -p $test_dir
        python wenet/bin/recognize_onnx.py --gpu 0 \
            --mode $mode \
            --config $dir/train.yaml \
            --encoder_onnx $dir/encoder.onnx \
            --decoder_onnx $dir/decoder.onnx \
            --test_data $test_feat_dir/format.data \
            --batch_size 20 \
            --dict $dict \
            --result_file $test_dir/text
        python tools/compute-wer.py --char=1 --v=1 \
            $test_feat_dir/text $test_dir/text > $test_dir/wer
    } &
    done
    wait
fi