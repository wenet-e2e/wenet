#!/usr/bin/env python3
#
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
"""
Usage:
export CUDA_VISIBLE_DEVICES="0"
python3 export_trt.py \
    --fp16 \
    --onnxFile exp6_fp16/encoderV4.onnx \
    --chunk_xs 1x67x80,32x67x80,64x67x80,1x67x80,4x67x80,8x67x80 \
    --chunk_lens 1,32,64,1,4,8 \
    --offset 1x1,32x1,64x1,1x1,4x1,8x1 \
    --att_cache 1x12x4x80x128,32x12x4x80x128,64x12x4x80x128,1x12x4x80x128,4x12x4x80x128,8x12x4x80x128 \ # noqa
    --cnn_cache 1x12x256x7,32x12x256x7,64x12x256x7,1x12x256x7,4x12x256x7,8x12x256x7 \
    --cache_mask 1x1x80,32x1x80,64x1x80,1x1x80,4x1x80,8x1x80 \
    --plugin exp6_fp16/LayerNorm.so \
    --trtFile exp6_fp16/encoder_test.plan \
    --test

"""

import ctypes
from cuda import cudart
import argparse
import numpy as np
import os
import torch
import tensorrt as trt
import timeit


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--chunk_xs",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--chunk_lens",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--offset",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--att_cache",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--cnn_cache",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--cache_mask",
        type=str,
        default="",
        help="BxTxD,BxTxD,BxTxD;BxTxD,BxTxD,BxTxD",
    )

    parser.add_argument(
        "--plugin",
        type=str,
        default="./LayerNorm.so",
        help="Path to the LayerNorm plugin",
    )

    parser.add_argument(
        "--timeCacheFile",
        type=str,
        default="./encoder.cache",
        help="Path to the saved engine cache file",
    )

    parser.add_argument(
        "--trtFile",
        type=str,
        default="./encoder.plan",
        help="Path to the exported tensorrt engine",
    )

    parser.add_argument(
        "--onnxFile",
        type=str,
        default="./encoder.onnx",
        help="Path to the onnx file",
    )

    parser.add_argument(
        "--useTimeCache",
        action="store_true",
        help="whether to use time cache, default false",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to export fp16 model, default false",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to test exported engine, default false",
    )

    return parser


def get_latency_result(latency_list, batch_size):
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)
    throughput_trt = 1000.0 / latency_ms

    return {
        "test_times":
        len(latency_list),
        "latency_variance":
        "{:.2f}".format(latency_variance),
        "latency_90_percentile":
        "{:.2f}".format(np.percentile(latency_list, 90) * 1000.0),
        "latency_95_percentile":
        "{:.2f}".format(np.percentile(latency_list, 95) * 1000.0),
        "latency_99_percentile":
        "{:.2f}".format(np.percentile(latency_list, 99) * 1000.0),
        "average_latency_ms":
        "{:.2f}".format(latency_ms),
        "QPS":
        "{:.2f}".format(throughput),
        f"QPS_trt_batch{batch_size}":
        "{:.2f}".format(throughput_trt),
    }


def test(engine, context, nBatchSize, batch_threshold=8):
    nProfile = engine.num_optimization_profiles
    if nProfile == 1:
        bindingBias = 0
    else:
        if nBatchSize > batch_threshold:
            bindingBias = 0
            context.set_optimization_profile_async(0, 0)
            cudart.cudaStreamSynchronize(0)
        else:
            bindingBias = int(engine.num_bindings / nProfile)
            context.set_optimization_profile_async(1, 0)
            cudart.cudaStreamSynchronize(0)

    context.set_binding_shape(bindingBias, [nBatchSize, 67, 80])
    context.set_binding_shape(bindingBias + 1, [nBatchSize])
    context.set_binding_shape(bindingBias + 2, [nBatchSize, 1])
    context.set_binding_shape(bindingBias + 3, [nBatchSize, 12, 4, 80, 128])
    context.set_binding_shape(bindingBias + 4, [nBatchSize, 12, 256, 7])
    context.set_binding_shape(bindingBias + 5, [nBatchSize, 1, 80])

    nInput = np.sum(
        [engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput

    nInput = nInput // nProfile
    nOutput = nOutput // nProfile

    chunk_xs = torch.randn(nBatchSize, 67, 80, dtype=torch.float32).numpy()
    chunk_lens = 67 * torch.ones(nBatchSize, dtype=torch.int32).numpy()

    offset = torch.arange(0, nBatchSize).unsqueeze(1).numpy()
    #  (elayers, b, head, cache_t1, d_k * 2)
    head = 4
    d_k = 64
    att_cache = torch.randn(nBatchSize,
                            12,
                            head,
                            80,
                            d_k * 2,
                            dtype=torch.float32).numpy()
    cnn_cache = torch.randn(nBatchSize, 12, 256, 7, dtype=torch.float32)

    cache_mask = torch.ones(nBatchSize, 1, 80, dtype=torch.float32)

    input_tensors = [
        chunk_xs,
        chunk_lens,
        offset,
        att_cache,
        cnn_cache,
        cache_mask,
    ]
    bufferH = []
    for data in input_tensors:
        bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(
            np.empty(
                context.get_binding_shape(bindingBias + i),
                dtype=trt.nptype(engine.get_binding_dtype(bindingBias + i)),
            ))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    if nProfile == 1 or nBatchSize > batch_threshold:
        bufferD = bufferD + [int(0) for _ in range(bindingBias)]
    else:
        bufferD = [int(0) for _ in range(bindingBias)] + bufferD

    for i in range(nInput):
        cudart.cudaMemcpy(
            bufferD[i],
            bufferH[i].ctypes.data,
            bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

    nWarm, nTest = 5, 10
    timeit.repeat(lambda: context.execute_v2(bufferD), number=1,
                  repeat=nWarm)  # Dry run
    latency_list = timeit.repeat(lambda: context.execute_v2(bufferD),
                                 number=1,
                                 repeat=nTest)
    print(get_latency_result(latency_list, nBatchSize))

    if nProfile == 1 or nBatchSize > batch_threshold:
        bufferD = bufferD[:bindingBias]
    else:
        bufferD = bufferD[-bindingBias:]

    for b in bufferD:
        cudart.cudaFree(b)


def main():
    args = get_parser().parse_args()

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    ctypes.cdll.LoadLibrary(args.plugin)

    timeCache = b""
    if args.useTimeCache and os.path.isfile(args.timeCacheFile):
        with open(args.timeCacheFile, "rb") as f:
            timeCache = f.read()
        if timeCache is None:
            print("Failed getting serialized timing cache!")
            exit()
        print("Succeeded getting serialized timing cache!")

    if os.path.isfile(args.trtFile):
        print("Engine existed!")
        with open(args.trtFile, "rb") as f:
            engineString = f.read()
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        if args.useTimeCache:
            cache = config.create_timing_cache(timeCache)
            config.set_timing_cache(cache, False)

        if args.fp16:
            config.flags = 1 << int(trt.BuilderFlag.FP16)
        # config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
        # config.flags = config.flags | (1 << int(trt.BuilderFlag.DEBUG))
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(args.onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(args.onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing ONNX file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing ONNX file!")

        profile0 = builder.create_optimization_profile()
        profile1 = builder.create_optimization_profile()
        for key, value in vars(args).items():
            if isinstance(value, str) and "," in value:
                shapes = [
                    tuple(map(int, item.split("x")))
                    for item in value.split(",")
                ]
                assert len(shapes) == 2 * 3
                profile0.set_shape(key, shapes[0], shapes[1], shapes[2])
                profile1.set_shape(key, shapes[3], shapes[4], shapes[5])

        config.add_optimization_profile(profile0)
        config.add_optimization_profile(profile1)

        engineString = builder.build_serialized_network(network, config)

        if engineString is None:
            print("Failed building engine!")
            exit()

        if args.useTimeCache and not os.path.isfile(args.timeCacheFile):
            timeCache = config.get_timing_cache()
            timeCacheString = timeCache.serialize()
            with open(args.timeCacheFile, "wb") as f:
                f.write(timeCacheString)
                print("Succeeded saving .cache file!")

        print("Succeeded building engine!")
        with open(args.trtFile, "wb") as f:
            f.write(engineString)

    if args.test:
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        context = engine.create_execution_context()
        batch_sizes = [1, 4, 8, 16, 32]
        for batch in batch_sizes:
            test(engine, context, batch)


if __name__ == "__main__":
    main()
