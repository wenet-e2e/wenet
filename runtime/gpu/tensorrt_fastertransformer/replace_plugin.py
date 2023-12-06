#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#

import onnx
import onnx_graphsurgeon as gs
import argparse
import utils


@gs.Graph.register()
def replace_plugin(self, inputs, outputs, op, name, attrs):

    for out in outputs:
        out.inputs.clear()

    return self.layer(op=op,
                      inputs=inputs,
                      outputs=outputs,
                      name=name,
                      attrs=attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='process onnx file for trt engine generation')
    parser.add_argument('--input_onnx',
                        type=str,
                        required=True,
                        help="input onnx model path")
    parser.add_argument('--output_onnx',
                        type=str,
                        required=True,
                        help="output .npy file path")
    parser.add_argument('--max_len',
                        type=int,
                        default=5000,
                        help="Max seq for pos embedding, TODO: remove this")
    parser.add_argument('--head_num',
                        type=int,
                        default=4,
                        choices=[4, 8],
                        help="")
    parser.add_argument('--feature_size', type=int, default=80, help="")
    parser.add_argument('--inter_size', type=int, default=2048, help="")
    parser.add_argument('--d_model',
                        type=int,
                        default=256,
                        choices=[256, 512],
                        help="")
    parser.add_argument('--num_layer', type=int, default=12, help="")
    parser.add_argument('--vocab_size', type=int, default=4233, help="")
    parser.add_argument('--conv_module_kernel_size',
                        type=int,
                        default=15,
                        choices=[15, 31],
                        help="kernel size for conv module")
    # TODO: hard-coding below encoder decoder weight path, pls don't change it for now
    parser.add_argument('--decoder_weight_path',
                        type=str,
                        default="/weight/dec/",
                        help="decoder weights path")
    parser.add_argument('--encoder_weight_path',
                        type=str,
                        default="/weight/enc/",
                        help="encoder weights path")
    parser.add_argument('--useFP16',
                        type=bool,
                        default=True,
                        help="using fp16 mode")
    parser.add_argument('--use_layernorm_in_conv_module',
                        action='store_true',
                        default=False,
                        help="using layernorm in conformer conv module")
    parser.add_argument('--q_scaling',
                        type=float,
                        default=1.0,
                        help="please hard-coding it for now")

    args = parser.parse_args()

    onnx_model = onnx.load(args.input_onnx)
    graph = gs.import_onnx(onnx_model)
    tmap = graph.tensors()

    if 'encoder' in args.input_onnx:
        inputs = [tmap[i] for i in ["speech", "speech_lengths"]]
        outputs = [
            tmap[i]
            for i in ["encoder_out", "encoder_out_lens", "ctc_log_probs"]
        ]
        op = "WenetEncoderPlugin"
        name = "WenetEncoder"
        attrs = {
            "max_len": 5000,  # TODO remove this, for pos embedding
            "head_num": args.head_num,
            "size_per_head": int(args.d_model / args.head_num),
            "feature_size": args.feature_size,
            "inter_size": args.inter_size,
            "d_model": args.d_model,
            "num_layer": args.num_layer,
            "use_layernorm_in_conv_module": args.use_layernorm_in_conv_module,
            "useFP16": args.useFP16,
            "vocab_size": args.vocab_size,
            "conv_module_kernel_size": args.conv_module_kernel_size,
            "weightFilePath": args.encoder_weight_path,
            "q_scaling": args.q_scaling,
        }

    elif 'decoder' in args.input_onnx:
        inputs = [
            tmap[i] for i in [
                "hyps_pad_sos_eos", "hyps_lens_sos", "encoder_out",
                "encoder_out_lens", "ctc_score"
            ]
        ]
        outputs = [tmap[i] for i in ["decoder_out", "best_index"]]
        op = "WenetDecoderPlugin"
        name = "WenetDecoder"
        attrs = {
            "head_num": args.head_num,
            "size_per_head": int(args.d_model / args.head_num),
            "inter_size": args.inter_size,
            "d_model": args.d_model,
            "num_layer": args.num_layer,
            "useFP16": args.useFP16,
            "vocab_size": args.vocab_size,
            "weightFilePath": args.decoder_weight_path,
            "q_scaling": args.q_scaling,
        }

    else:
        raise NotImplementedError

    graph.replace_plugin(inputs, outputs, op, name, attrs)

    graph.cleanup().toposort()
    utils.fold_const(graph)

    onnx.save(gs.export_onnx(graph), args.output_onnx)
