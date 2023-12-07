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

import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
import pycuda.autoinit  # noqa
import pycuda.driver as cuda

useFile = False
ipnutDataFile = "./layerNormIO-bs64.npz"
soFilePath = "./LayerNorm.so"
nBS = 1024
nSL = 256
nEmbedding = 256
nTime = 100
epsilon = 1e-6

np.random.seed(97)

npToTRT = {
    np.int8: trt.int8,
    np.float16: trt.float16,
    np.int32: trt.int32,
    np.float32: trt.float32,
}
npToPFT = {
    np.int8: trt.PluginFieldType.INT8,
    np.float16: trt.PluginFieldType.FLOAT16,
    np.int32: trt.PluginFieldType.INT32,
    np.float32: trt.PluginFieldType.FLOAT32,
}


def check(a, b, weak=False):
    if weak:
        return np.all(np.abs(a - b) < epsilon)
    else:
        return np.all(a == b)


def layerNormCPU(bufferH):
    _x, b, a = bufferH
    nEmbed = bufferH[0].shape[2]
    _0 = np.mean(_x, 2)[:, :, np.newaxis]
    _1 = _x - _0
    _2 = _1 * _1
    _3 = np.mean(_2, 2)[:, :, np.newaxis]
    _4 = np.array(1e-12, dtype=np.float32)
    _5 = _4.reshape(1, 1, 1)
    _6 = _3 + _5
    _7 = np.sqrt(_6)
    _8 = 1 / _7  # 1/sqrt(...)
    _9 = b
    _10 = _9.reshape(1, 1, nEmbed)
    _11 = _8 * _10  # b/sqrt(...)
    _12 = _0 * _11  # bμ/sqrt(...)
    _13 = a
    _14 = _13.reshape(1, 1, nEmbed)
    _15 = _14 - _12  # a-bμ/sqrt(...)
    _16 = _x * _11  # bx/sqrt(...)
    _17 = _15 + _16  # b(x-μ)/sqrt(...)+a
    _18 = _17.reshape(bufferH[0].shape[0], bufferH[0].shape[1],
                      bufferH[0].shape[2])
    return _18


def testLayerNormCPU():
    print("test LayerNormCPU!")
    bufferH = []
    io = np.load(ipnutDataFile)
    bufferH.append(io["encoder1_inputs:0"])
    bufferH.append(io["(Unnamed Layer* 9) [Constant]_output"])
    bufferH.append(io["(Unnamed Layer* 13) [Constant]_output"])

    temp1 = layerNormCPU(bufferH)
    print("outputCPU: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f" % (
        str(temp1.shape),
        np.sum(abs(temp1)),
        np.var(temp1),
        np.max(temp1),
        np.min(temp1),
        np.sum(np.abs(np.diff(temp1.reshape(-1)))),
    ))
    # print(temp1)
    temp2 = io[
        "seq2seq/encoder_1/layer_0/multi_head/conv1d/conv1d/ExpandDims:0"]
    print("outputRef: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f" % (
        str(temp2.shape),
        np.sum(abs(temp2)),
        np.var(temp2),
        np.max(temp2),
        np.min(temp2),
        np.sum(np.abs(np.diff(temp2.reshape(-1)))),
    ))
    # print(temp2)
    print("check result:")
    print(check(temp1, temp2, True))
    print("test layerNormCPU finish!")


def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        if c.name == "LayerNorm":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None


def run():
    testCase = "test<fp%s,bs=%d,sl=%d,nEmbed=%d>" % (
        ["32", "16"][0],
        nBS,
        nSL,
        nEmbedding,
    )
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    ctypes.cdll.LoadLibrary(soFilePath)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << 0)
    config = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags = [0, 1 << int(trt.BuilderFlag.FP16)][0]

    inputTensorList = []
    inputTensorList.append(
        network.add_input("inputT", trt.float32, [-1, -1, 256]))
    inputTensorList.append(network.add_input("inputB", trt.float32, [256]))
    inputTensorList.append(network.add_input("inputA", trt.float32, [256]))

    profile = builder.create_optimization_profile()
    profile.set_shape("inputT", [1, 4, 256], [1024, 256, 256],
                      [1024, 256, 256])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    pluginLayer.get_output(0).dtype = [trt.float32, trt.float16][0]

    network.mark_output(pluginLayer.get_output(0))

    engine = builder.build_engine(network, config)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBS, nSL, nEmbedding])
    context.set_binding_shape(1, [nEmbedding])
    context.set_binding_shape(2, [nEmbedding])
    print("Binding all? %s" %
          (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    stream = cuda.Stream()

    nInput = np.sum(
        [engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print(
            "input ->" if engine.binding_is_input(i) else "output->",
            engine.get_binding_dtype(i),
            engine.get_binding_shape(i),
            context.get_binding_shape(i),
        )

    bufferH = []
    bufferH.append(
        np.random.rand(nBS, nSL, nEmbedding).astype(np.float32).reshape(
            nBS, nSL, nEmbedding) * 2 - 1)
    bufferH.append(np.ones(nEmbedding).astype(np.float32))
    bufferH.append(np.zeros(nEmbedding).astype(np.float32))
    bufferH.append(
        np.empty(
            context.get_binding_shape(3),
            dtype=trt.nptype(engine.get_binding_dtype(3)),
        ))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cuda.mem_alloc(bufferH[i].nbytes))

    for i in range(nInput):
        cuda.memcpy_htod_async(bufferD[i],
                               np.ascontiguousarray(bufferH[i].reshape(-1)),
                               stream)

    context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()

    for i in range(nOutput):
        cuda.memcpy_dtoh_async(bufferH[nInput + i], bufferD[nInput + i],
                               stream)
    stream.synchronize()

    for i in range(nInput):
        temp = bufferH[i]
        print(
            "inputH%d" % i,
            temp.shape,
            np.sum(abs(temp)),
            np.var(temp),
            np.max(temp),
            np.min(temp),
            np.sum(np.abs(np.diff(temp.reshape(-1)))),
        )
        print(temp.reshape(-1)[:10])
        # print(temp)

    for i in range(nOutput):
        temp = bufferH[nInput + i]
        print(
            "outputH%d" % i,
            temp.shape,
            np.sum(abs(temp)),
            np.var(temp),
            np.max(temp),
            np.min(temp),
            np.sum(np.abs(np.diff(temp.reshape(-1)))),
        )
        # print(temp)

    for i in range(10):
        context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()

    time0 = time_ns()
    for i in range(nTime):
        context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()
    time1 = time_ns()
    print(testCase + "average %fms per inference\n" %
          ((time1 - time0) / nTime / 1000000))

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])

    print(
        check(temp1, temp2, True),
        "max diff=%f" % (np.max(np.abs(temp1 - temp2))),
    )


if __name__ == "__main__":
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    run()

    # print("test all finish!")
