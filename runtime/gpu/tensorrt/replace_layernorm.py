# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="process onnx file for trt engine generation")
    parser.add_argument("--input_onnx",
                        type=str,
                        required=True,
                        help="input onnx model path")
    parser.add_argument("--output_onnx",
                        type=str,
                        required=True,
                        help="output .npy file path")
    args = parser.parse_args()

    sourceOnnx = args.input_onnx
    destinationOnnx = args.output_onnx

    graph = gs.import_onnx(
        onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

    nLayerNormPlugin = 0
    for node in graph.nodes:
        if (node.op == "ReduceMean" and node.o().op == "Sub"
                and node.o().inputs[0] == node.inputs[0]
                and node.o().o(0).op == "Pow" and node.o().o(1).op == "Div"
                and node.o().o(0).o().op == "ReduceMean"
                and node.o().o(0).o().o().op == "Add"
                and node.o().o(0).o().o().o().op == "Sqrt"
                and node.o().o(0).o().o().o().o().op == "Div"
                and node.o().o(0).o().o().o().o() == node.o().o(1)
                and node.o().o(0).o().o().o().o().o().op == "Mul"
                and node.o().o(0).o().o().o().o().o().o().op == "Add"):
            inputTensor = node.inputs[0]

            lastMultipyNode = node.o().o(0).o().o().o().o().o()
            index = ["weight" in i.name
                     for i in lastMultipyNode.inputs].index(True)
            b = np.array(
                deepcopy(lastMultipyNode.inputs[index].values.tolist()),
                dtype=np.float32,
            )
            # MUST use np.ascontiguousarray,
            # or TRT will regard the shape of this Constant as (0) !!!
            constantB = gs.Constant(
                "LayerNormB-" + str(nLayerNormPlugin),
                np.ascontiguousarray(b.reshape(-1)),
            )

            lastAddNode = node.o().o(0).o().o().o().o().o().o()
            index = ["bias" in i.name for i in lastAddNode.inputs].index(True)
            a = np.array(
                deepcopy(lastAddNode.inputs[index].values.tolist()),
                dtype=np.float32,
            )
            constantA = gs.Constant(
                "LayerNormA-" + str(nLayerNormPlugin),
                np.ascontiguousarray(a.reshape(-1)),
            )

            inputList = [inputTensor, constantB, constantA]
            layerNormV = gs.Variable(
                "LayerNormV-" + str(nLayerNormPlugin),
                np.dtype(np.float32),
                None,
            )
            layerNormN = gs.Node(
                "LayerNorm",
                "LayerNormN-" + str(nLayerNormPlugin),
                inputs=inputList,
                outputs=[layerNormV],
            )
            graph.nodes.append(layerNormN)

            # the last LayerNorm provide one of the graph's output,
            # and do not unsqueeze to 4 dimension
            if lastAddNode.outputs[0] in graph.outputs:
                # oldLastAdd -> graph.outputs[0] ===>
                # LayerNorm -> Squeeze -> graph.outputs[0]
                layerNormN.outputs[0].name = "chunk_out"
                index = graph.outputs.index(lastAddNode.outputs[0])
                # TODO: FIX ME YUEKAI, for offline asr encoder_out dtype
                graph.outputs[index] = layerNormN.outputs[0].to_variable(
                    np.float16)
                # graph.outputs[index] = layerNormN.outputs[0]
            else:  # other LayerNorm contain the subsequent Squeeze operation
                for n in graph.nodes:
                    if lastAddNode.outputs[0] in n.inputs:
                        index = n.inputs.index(lastAddNode.outputs[0])
                        n.inputs[index] = layerNormN.outputs[0]

                lastAddNode.outputs = []
            nLayerNormPlugin += 1
            continue

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), destinationOnnx)
