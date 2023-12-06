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
import numpy as np


def get_parent(graph, cur_node, parrent_pos):
    in0 = cur_node.inputs[parrent_pos]
    for node in graph.nodes:
        for out in node.outputs:
            if out == in0:
                return node
    return None


def get_weight_by_bias(graph, bname):
    for node in graph.nodes:
        if node.op == "Add" and node.inputs[0].name == bname:
            pnode = get_parent(graph, node, 1)
            if pnode is not None and pnode.op == "MatMul":
                return pnode.inputs[1].name
    return None


def onnx_GetAllWeight(model):
    for w in model.graph.initializer:
        print(w.name, w.dims)
    return model.graph.initializer


def onnx2np_type(dtype):
    maps = {1: np.float32, 6: np.int32, 7: np.int64}
    return maps[dtype]


def onnx_GetWeight(model, name):
    for w in model.graph.initializer:
        if w.name == name:
            return w
    return None


def fold_const(graph):
    old_cc = len(graph.nodes)
    graph.fold_constants()
    graph.cleanup().toposort()
    new_cc = len(graph.nodes)
    print("fold const:", old_cc, " -> ", new_cc, " = ", old_cc - new_cc)
