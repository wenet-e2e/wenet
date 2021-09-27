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

# dir is model directory, which can be
# downloaded from https://github.com/wenet-e2e/wenet/tree/main/examples/aishell2/s0
# if you don't have one
# Please edit ctc weight, reverse weight, global_cmvn path in config.yaml or train.yaml

dir=20210618_u2pp_conformer_exp
beam_size=10

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        python3 -c 'import onnxruntime' || exit 1;
        echo "export to offline onnx"
        python3 wenet/bin/export_onnx.py --config $dir/train.yaml \
            --checkpoint $dir/final.pt \
            --beam_size $beam_size \
            --output_onnx_directory $dir
fi

