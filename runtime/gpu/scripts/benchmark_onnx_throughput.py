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
# Modified from below:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/onnx_exporter.py
"""
Usage:
export CUDA_VISIBLE_DEVICES="0"
python3 test_onnx_throughput.py \
      --batch_sizes 1,4,16 \
      --sequence_lenghts 67 \
      --onnxFile ./encoder.onnx \
      --model_type streaming_conformer_encoder \
      --log ./log.txt

"""

import timeit
import onnxruntime
import torch
import argparse
import numpy


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,4,32",
        help="batch sizes for infer e.g. 1,2,4,8",
    )

    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="67",
        help="sequence frames for infer e.g. 500,800,2000",
    )

    parser.add_argument(
        "--onnxFile",
        type=str,
        default="wenet/bin/u2pp_aishell2_onnx/encoder_fp16.onnx",
        help="Path to the onnx file",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="./log.txt",
        help="Path to the log file",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "streaming_conformer_encoder",
            "conformer_encoder",
            "decoder",
            "bidecoder",
        ],
        default="streaming_conformer_encoder",
        help="onnx model type for wenet",
    )

    parser.add_argument(
        "--disable_gpu",
        action="store_true",
        help="whether to disable gpu infer, default false",
    )

    parser.add_argument(
        "--disable_ort_io_binding",
        action="store_true",
        help="whether to disable onnxrt io binding",
    )

    return parser


def allocateOutputBuffers(output_buffers,
                          output_buffer_max_sizes,
                          device,
                          data_type=torch.float32):
    # Allocate output tensors with the largest test size needed.
    # So the allocated memory can be reused
    # for each test run.

    for i in output_buffer_max_sizes:
        output_buffers.append(torch.empty(i, dtype=data_type, device=device))


def get_latency_result(latency_list, batch_size):
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = numpy.var(latency_list, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)
    throughput_trt = 1000.0 / latency_ms

    return {
        "test_times":
        len(latency_list),
        "latency_variance":
        "{:.2f}".format(latency_variance),
        "latency_90_percentile":
        "{:.2f}".format(numpy.percentile(latency_list, 90) * 1000.0),
        "latency_95_percentile":
        "{:.2f}".format(numpy.percentile(latency_list, 95) * 1000.0),
        "latency_99_percentile":
        "{:.2f}".format(numpy.percentile(latency_list, 99) * 1000.0),
        "average_latency_ms":
        "{:.2f}".format(latency_ms),
        "QPS":
        "{:.2f}".format(throughput),
        f"QPS_trt_batch{batch_size}":
        "{:.2f}".format(throughput_trt),
    }


def create_onnxruntime_input(
    batch_size,
    sequence_length,
    input_names,
    config,
    model_type,
    data_type=torch.float16,
):
    inputs = {}
    if model_type == "streaming_conformer_encoder":
        feature_size = 80
        num_layers = 12
        head = 4
        required_cache_size = 80
        output_size = 256
        d_k = int(output_size / head)
        cnn_module_kernel = 7

        chunk_xs = torch.randn(batch_size,
                               sequence_length,
                               feature_size,
                               dtype=data_type).numpy()
        inputs["chunk_xs"] = chunk_xs
        chunk_lens = (torch.ones(batch_size, dtype=torch.int32).numpy() *
                      sequence_length)
        inputs["chunk_lens"] = chunk_lens
        offset = (torch.arange(0, batch_size,
                               dtype=torch.int64).unsqueeze(1).numpy())
        inputs["offset"] = offset
        att_cache = torch.randn(
            batch_size,
            num_layers,
            head,
            required_cache_size,
            d_k * 2,
            dtype=data_type,
        ).numpy()
        inputs["att_cache"] = att_cache
        cnn_cache = torch.randn(
            batch_size,
            num_layers,
            output_size,
            cnn_module_kernel,
            dtype=data_type,
        ).numpy()
        inputs["cnn_cache"] = cnn_cache
        cache_mask = torch.ones(batch_size,
                                1,
                                required_cache_size,
                                dtype=data_type).numpy()
        inputs["cache_mask"] = cache_mask

    else:
        return NotImplementedError
    return inputs


def inference_ort(
    ort_session,
    ort_inputs,
    result_template,
    repeat_times,
    batch_size,
    warm_up_repeat=0,
):
    result = {}
    timeit.repeat(
        lambda: ort_session.run(None, ort_inputs),
        number=1,
        repeat=warm_up_repeat,
    )  # Dry run
    latency_list = timeit.repeat(lambda: ort_session.run(None, ort_inputs),
                                 number=1,
                                 repeat=repeat_times)
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(latency_list, batch_size))
    return result


IO_BINDING_DATA_TYPE_MAP = {
    "float32": numpy.float32,
    "float16": numpy.float16,
    "int32": numpy.int32,
    "int64": numpy.int64
    # TODO: Add more.
}


def inference_ort_with_io_binding(
    ort_session,
    ort_inputs,
    result_template,
    repeat_times,
    ort_output_names,
    ort_outputs,
    output_buffers,
    output_buffer_max_sizes,
    batch_size,
    device,
    data_type=numpy.longlong,
    warm_up_repeat=0,
):
    result = {}

    # Bind inputs and outputs to onnxruntime session
    io_binding = ort_session.io_binding()
    # Bind inputs to device
    for name in ort_inputs.keys():
        np_input = torch.from_numpy(ort_inputs[name]).to(device)
        input_type = (IO_BINDING_DATA_TYPE_MAP[str(ort_inputs[name].dtype)]
                      if str(ort_inputs[name].dtype)
                      in IO_BINDING_DATA_TYPE_MAP else data_type)

        io_binding.bind_input(
            name,
            np_input.device.type,
            0,
            input_type,
            np_input.shape,
            np_input.data_ptr(),
        )
    # Bind outputs buffers with the sizes needed if not allocated already
    if len(output_buffers) == 0:
        allocateOutputBuffers(output_buffers, output_buffer_max_sizes, device)

    for i, ort_output_name in enumerate(ort_output_names):
        output_type = (IO_BINDING_DATA_TYPE_MAP[str(ort_outputs[i].dtype)]
                       if str(ort_outputs[i].dtype) in IO_BINDING_DATA_TYPE_MAP
                       else data_type)
        io_binding.bind_output(
            ort_output_name,
            output_buffers[i].device.type,
            0,
            output_type,
            ort_outputs[i].shape,
            output_buffers[i].data_ptr(),
        )

    timeit.repeat(
        lambda: ort_session.run_with_iobinding(io_binding),
        number=1,
        repeat=warm_up_repeat,
    )  # Dry run

    latency_list = timeit.repeat(
        lambda: ort_session.run_with_iobinding(io_binding),
        number=1,
        repeat=repeat_times,
    )
    result.update(result_template)
    result.update({"io_binding": True})
    result.update(get_latency_result(latency_list, batch_size))
    return result


def create_onnxruntime_session(
    onnx_model_path,
    use_gpu,
    provider=None,
    enable_all_optimization=True,
    num_threads=-1,
    enable_profiling=False,
    verbose=False,
    provider_options=None,
):
    session = None
    sess_options = onnxruntime.SessionOptions()

    if enable_all_optimization:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
    else:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL # noqa

    if enable_profiling:
        sess_options.enable_profiling = True

    if num_threads > 0:
        sess_options.intra_op_num_threads = num_threads

    if verbose:
        sess_options.log_severity_level = 0
    else:
        sess_options.log_severity_level = 4

    if use_gpu:
        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "tensorrt":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    if provider_options:
        providers = [
            (name,
             provider_options[name]) if name in provider_options else name
            for name in providers
        ]

    session = onnxruntime.InferenceSession(onnx_model_path,
                                           sess_options,
                                           providers=providers)

    return session


if __name__ == "__main__":
    args = get_parser().parse_args()
    warm_up_repeat = 20
    repeat_times = 20
    input_value_type = torch.float16

    batch_sizes = list(map(int, args.batch_sizes.split(",")))
    max_sequence_length = None
    sequence_lengths = list(map(int, args.sequence_lengths.split(",")))

    if args.model_type == "streaming_conformer_encoder":
        input_names = [
            "chunk_xs",
            "chunk_lens",
            "offset",
            "att_cache",
            "cnn_cache",
            "cache_mask",
        ]
        ort_output_names = [
            "log_probs",
            "log_probs_idx",
            "chunk_out",
            "chunk_out_lens",
            "r_offset",
            "r_att_cache",
            "r_cnn_cache",
            "r_cache_mask",
        ]
    else:
        raise NotImplementedError

    if args.disable_gpu:
        EP_list = ["CPUExecutionProvider"]
    else:
        EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    device = "cpu" if args.disable_gpu else "cuda"

    ort_session = create_onnxruntime_session(
        args.onnxFile,
        not args.disable_gpu,
        provider="cuda",
        enable_all_optimization=True,
        num_threads=-1,
        verbose=False,
    )

    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            if (max_sequence_length is not None
                    and sequence_length > max_sequence_length):
                continue

            ort_inputs = create_onnxruntime_input(
                batch_size,
                sequence_length,
                input_names,
                input_value_type,
                args.model_type,
            )

            result_template = {
                "io_binding": not args.disable_ort_io_binding,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
            }

            print("Run onnxruntime on {} with input shape {}".format(
                args.onnxFile, [batch_size, sequence_length]))

            if args.disable_ort_io_binding:
                result = inference_ort(
                    ort_session,
                    ort_inputs,
                    result_template,
                    repeat_times,
                    batch_size,
                    warm_up_repeat,
                )
            else:
                # Get output sizes from a dummy ort run
                ort_outputs = ort_session.run(ort_output_names, ort_inputs)
                output_buffer_max_sizes = []
                for i in range(len(ort_outputs)):
                    output_buffer_max_sizes.append(
                        numpy.prod(ort_outputs[i].shape))

                data_type = numpy.intc
                output_buffers = []
                result = inference_ort_with_io_binding(
                    ort_session,
                    ort_inputs,
                    result_template,
                    repeat_times,
                    ort_output_names,
                    ort_outputs,
                    output_buffers,
                    output_buffer_max_sizes,
                    batch_size,
                    device,
                    data_type,
                    warm_up_repeat,
                )
            print(result)
            results.append(result)
    with open(args.log, "w") as log_f:
        for result in results:
            log_f.write(f"{result}\n")
