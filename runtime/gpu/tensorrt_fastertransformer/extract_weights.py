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

import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

import utils


def get_not(model, exported_name):
    not_name = list()
    for w in model.graph.initializer:
        if w.name not in exported_name:
            not_name.append(w.name)
    return not_name


def export_GetAllWeight(model, gsg):
    exported_name = list()
    res = dict()
    for w in model.graph.initializer:
        if 'encoder' in str(w.name) or 'ctc' in str(w.name):
            print("export ", w.name, w.dims, w.data_type)
            dtype = utils.onnx2np_type(w.data_type)
            res[w.name] = np.frombuffer(w.raw_data,
                                        dtype=dtype).reshape(w.dims)
            exported_name.append(w.name)
            if w.name.endswith("bias"):
                new_name = w.name[0:len(w.name) - 4] + "weight"
                wname = utils.get_weight_by_bias(gsg, w.name)
                if wname is None:
                    continue
                w = utils.onnx_GetWeight(model, wname)
                dtype = utils.onnx2np_type(w.data_type)
                res[new_name] = np.frombuffer(w.raw_data,
                                              dtype=dtype).reshape(w.dims)
                res[new_name] = np.transpose(res[new_name], (1, 0))
                print("export ", w.name, w.dims, w.data_type, " -> ", new_name,
                      res[new_name].shape)
                exported_name.append(w.name)

    not_name = get_not(model, exported_name)
    cur_idx = 0
    for w in model.graph.initializer:
        if w.name in not_name and len(w.dims) == 2 \
           and (w.dims[0] == 256 or w.dims[0] == 512) \
           and (w.dims[1] == 256 or w.dims[1] == 512):
            for node in gsg.nodes:
                if node.op == "MatMul" and node.i(0, 0).op == "Slice" \
                   and node.inputs[1].name == w.name:
                    new_name = "encoder.encoders." + \
                        str(cur_idx) + ".self_attn.linear_pos.weight"
                    print("export ", w.name, w.dims, w.data_type, " -> ",
                          new_name)
                    dtype = utils.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data,
                                                  dtype=dtype).reshape(w.dims)
                    res[new_name] = np.transpose(res[new_name], (1, 0))
                    print("export ", w.name, w.dims, w.data_type, " -> ",
                          new_name, res[new_name].shape)
                    exported_name.append(w.name)
                    cur_idx += 1

    not_name = get_not(model, exported_name)
    cur_idx = 0
    for w in model.graph.initializer:
        if w.name in not_name and len(w.dims) == 3:
            for node in gsg.nodes:
                if node.op == "Conv" and len(node.inputs) == 3 \
                   and node.inputs[1].name == w.name:
                    new_name = "encoder.encoders." + \
                        str(cur_idx) + ".conv_module.depthwise_conv.weight"
                    print("export ", w.name, w.dims, w.data_type, " -> ",
                          new_name)
                    dtype = utils.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data,
                                                  dtype=dtype).reshape(w.dims)
                    exported_name.append(w.name)

                    bname = node.inputs[2].name
                    w = utils.onnx_GetWeight(model, bname)
                    new_name = "encoder.encoders." + \
                        str(cur_idx) + ".conv_module.depthwise_conv.bias"
                    print("export ", w.name, w.dims, w.data_type, " -> ",
                          new_name)
                    dtype = utils.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data,
                                                  dtype=dtype).reshape(w.dims)
                    exported_name.append(w.name)
                    cur_idx += 1

    for node in gsg.nodes:
        if node.op == "Slice" and 'value' in node.i(0, 0).attrs:
            pnode = node.i(0, 0)
            print(node.name)
            w = pnode.attrs["value"]
            new_name = "encoder.positional.encoding.data"
            print("export ", w.name, w.shape, w.dtype, " -> ", new_name)
            print(type(w.values))
            res[new_name] = w.values

    assert "encoder.positional.encoding.data" in res

    return res


def export_decoder_GetAllWeight(model, gsg):
    exported_name = list()
    res = dict()
    for w in model.graph.initializer:
        if len(str(w.name)) > 4:
            print("export ", w.name, w.dims, w.data_type)
            dtype = utils.onnx2np_type(w.data_type)
            res[w.name] = np.frombuffer(w.raw_data,
                                        dtype=dtype).reshape(w.dims)
            exported_name.append(w.name)
            if w.name.endswith("bias"):
                new_name = w.name[0:len(w.name) - 4] + "weight"
                wname = utils.get_weight_by_bias(gsg, w.name)
                if wname is None:
                    continue
                w = utils.onnx_GetWeight(model, wname)
                dtype = utils.onnx2np_type(w.data_type)
                res[new_name] = np.frombuffer(w.raw_data,
                                              dtype=dtype).reshape(w.dims)
                res[new_name] = np.transpose(res[new_name], (1, 0))
                print("export ", w.name, w.dims, w.data_type, " -> ", new_name,
                      res[new_name].shape)
                exported_name.append(w.name)

    for node in gsg.nodes:
        if node.op == "Slice" and 'value' in node.i(0, 0).attrs:
            pnode = node.i(0, 0)
            w = pnode.attrs["value"]
            print(w.values, w.values.shape)
            new_name = "decoder.positional.encoding.data"
            print("export ", w.name, w.shape, w.dtype, " -> ", new_name)
            print(type(w.values))
            res[new_name] = w.values

    not_name = get_not(model, exported_name)
    for w in model.graph.initializer:
        if w.name in not_name:
            dtype = utils.onnx2np_type(w.data_type)
            cur = np.frombuffer(w.raw_data, dtype=dtype).reshape(w.dims)
            print("not export ", w.name, w.dims, w.data_type, cur)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='process onnx file for trt engine generation')
    parser.add_argument('--input_onnx',
                        type=str,
                        required=True,
                        help="input onnx model path")
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help="output weights dir")

    args = parser.parse_args()

    onnx_model = onnx.load(args.input_onnx)
    graph = gs.import_onnx(onnx_model)

    if 'encoder' in args.input_onnx:
        result = export_GetAllWeight(onnx_model, graph)

    elif 'decoder' in args.input_onnx:
        result = export_decoder_GetAllWeight(onnx_model, graph)

    else:
        raise NotImplementedError

    for name in result:
        saved_path = args.output_dir + "/" + name + ".bin"
        cur = result[name]
        if name.endswith(".weight") and len(cur.shape) == 2 \
           and "decoder.embed.0.weight" not in name:
            cur = cur.transpose((1, 0))

        if name.endswith(".pointwise_conv1.weight"):
            cur = cur.transpose((1, 0, 2))

        if name.endswith(".pointwise_conv2.weight"):
            cur = cur.transpose((1, 0, 2))

        if name.endswith(".depthwise_conv.weight") and len(cur.shape) == 3:
            cur = cur.transpose((2, 1, 0))

        if name.endswith(".weight") and ".embed.conv." in name:
            cur = cur.transpose((0, 2, 3, 1))

        cur.tofile(saved_path)

    print("extract Wenet model weight finish!")
