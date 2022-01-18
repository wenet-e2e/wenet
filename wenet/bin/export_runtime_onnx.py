# Copyright 2021 Huya Inc. All Rights Reserved.
# Author: lizexuan@huya.com (Zexuan Li)
# Reference from https://github.com/Mashiro009/wenet-onnx.git

from __future__ import print_function

import argparse
import os
import sys

import torch
import yaml

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import numpy as np
try:
    import onnx
    import onnxruntime
except ImportError:
    print('Please install onnx onnxruntime!')
    sys.exit(1)


def export_conf(model, conf_path):
    w = open(conf_path, 'w')
    w.write(str(output_size) + "\n")
    w.write(str(num_blocks) + "\n")
    w.write(str(cnn_module_kernel) + "\n")
    w.write(str(model.subsampling_rate()) + "\n")
    w.write(str(model.right_context()) + "\n")
    w.write(str(model.sos_symbol()) + "\n")
    w.write(str(model.eos_symbol()) + "\n")
    w.write(('1' if model.is_bidirectional_decoder() else '0') + "\n")
    w.close()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


def output_encoder_conformer_onnx(encoder_model, encoder_model_path):
    subsampling_cache = torch.zeros(1, 1, output_size)
    elayers_output_cache = torch.zeros(num_blocks, 1, 1, output_size)
    conformer_cnn_cache = torch.zeros(num_blocks, 1,
                                      encoder_model._output_size,
                                      cnn_module_kernel)

    inputs = [torch.randn(1, 60 * (i + 1), 80) for i in range(5)]
    dummy_input = inputs[0]
    offset = torch.LongTensor(1)
    offset[0] = 1
    required_cache_size = torch.LongTensor(1)
    required_cache_size[0] = -1

    torch.onnx.export(
        encoder_model,
        (dummy_input, offset, required_cache_size, subsampling_cache,
         elayers_output_cache, conformer_cnn_cache),
        encoder_model_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=[
            'input', 'offset', 'required_cache_size', 'i1', 'i2', 'i3'
        ],
        output_names=['output', 'o1', 'o2', 'o3'],
        dynamic_axes={
            'input': [1],
            'i1': [1],
            'i2': [2],
            'output': [1],
            'o1': [1],
            'o2': [2]
        },
        verbose=True)
    print('export encoder model done')


def check_encoder_onnx_and_pytorch(encoder_model, torch_encoder_model,
                                   encoder_model_path):
    # following is test torch encoder's function forward_chunk_onnx code
    inputs = [torch.randn(1, 60 * (i + 1), 80) for i in range(5)]
    offset = torch.LongTensor(1)
    offset[0] = 1
    required_cache_size = torch.LongTensor(1)
    required_cache_size[0] = -1

    subsampling_cache = torch.zeros(1, 1, encoder_model._output_size)
    elayers_output_cache = torch.zeros(num_blocks, 1, 1,
                                       encoder_model._output_size)
    conformer_cnn_cache = torch.zeros(num_blocks, 1,
                                      encoder_model._output_size,
                                      cnn_module_kernel)
    chunk_onnx_outputs = []
    for i in range(5):
        dummy_input = inputs[i]
        (out, 
         subsampling_cache, 
         elayers_output_cache, 
         conformer_cnn_cache) = encoder_model(dummy_input, offset, 
                                              required_cache_size, 
                                              subsampling_cache, 
                                              elayers_output_cache, 
                                              conformer_cnn_cache)
        chunk_onnx_outputs.append(out)
        offset += out.size(1)

    offset = torch.tensor(0, dtype=torch.int64)
    required_cache_size = torch.tensor(-1, dtype=torch.int64)
    subsampling_cache = None
    elayers_output_cache = None
    conformer_cnn_cache = None
    torch_outputs = []

    for i in range(5):
        dummy_input = inputs[i]
        (out2,
         subsampling_cache, 
         elayers_output_cache, 
         conformer_cnn_cache) = torch_encoder_model(dummy_input, offset, 
                                                    required_cache_size, 
                                                    subsampling_cache,
                                                    elayers_output_cache, 
                                                    conformer_cnn_cache)
        torch_outputs.append(out2)
        offset += out2.size(1)

    onnx_model = onnx.load(encoder_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("encoder onnx_model check pass!")

    ort_session = onnxruntime.InferenceSession(encoder_model_path)
    print(encoder_model_path + " onnx model has " +
          str(len(ort_session.get_inputs())) + " args")

    required_cache_size = torch.LongTensor(1)
    required_cache_size[0] = 3 * 64
    subsampling_cache = torch.zeros(1, 1, output_size)
    elayers_output_cache = torch.zeros(num_blocks, 1, 1, output_size)
    conformer_cnn_cache = torch.zeros(num_blocks, 1,
                                      encoder_model._output_size,
                                      cnn_module_kernel)
    offset = torch.LongTensor(1)
    offset[0] = 1
    subsampling_cache = to_numpy(subsampling_cache)
    elayers_output_cache = to_numpy(elayers_output_cache)
    conformer_cnn_cache = to_numpy(conformer_cnn_cache)
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(dummy_input),
            ort_session.get_inputs()[1].name: to_numpy(offset),
            ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
            ort_session.get_inputs()[3].name: subsampling_cache,
            ort_session.get_inputs()[4].name: elayers_output_cache,
            ort_session.get_inputs()[5].name: conformer_cnn_cache
        }
        ort_outs = ort_session.run(None, ort_inputs)
        offset += ort_outs[0].shape[1]
        subsampling_cache = ort_outs[1]
        elayers_output_cache = ort_outs[2]
        conformer_cnn_cache = ort_outs[3]
        np.testing.assert_allclose(to_numpy(torch_outputs[i]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   to_numpy(torch_outputs[i]),
                                   rtol=1e-03,
                                   atol=1e-05)

    print("Exported encoder model has been tested with ONNXRuntime, \
and the result looks good!")


def output_ctc_onnx(ctc_model, ctc_model_path):

    inputs = [torch.randn(1, 60 * (i + 1), output_size) for i in range(5)]
    dummy_input = inputs[0]
    torch.onnx.export(ctc_model, (dummy_input),
                      ctc_model_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': [1],
                          'output': [1]},
                      verbose=True)
    print('export ctc model done')


def check_ctc_onnx_and_pytorch(ctc_model, torch_ctc_model, ctc_model_path):

    inputs = [torch.randn(1, 60 * (i + 1), output_size) for i in range(5)]
    chunk_onnx_outputs = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        out = torch_ctc_model(dummy_input)
        chunk_onnx_outputs.append(out)

    torch_outputs = []
    for i in range(5):
        dummy_input = inputs[i]
        out = ctc_model(dummy_input)
        torch_outputs.append(out)

    onnx_model = onnx.load(ctc_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("ctc onnx_model check pass!")

    ort_session = onnxruntime.InferenceSession(ctc_model_path)
    print(ctc_model_path + " onnx model has " +
          str(len(ort_session.get_inputs())) + " args")

    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(dummy_input),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(torch_outputs[i]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   to_numpy(torch_outputs[i]),
                                   rtol=1e-03,
                                   atol=1e-05)

    print("Exported ctc model has been tested with ONNXRuntime, \
and the result looks good!")


def output_rescore_onnx(rescore_model, rescore_model_path):
    inputs = [torch.randn(1, 60 * (i + 1), output_size) for i in range(5)]
    hyps_pad = (abs(torch.randn(10, 30)) * 1000).ceil().long()

    hyps_lens = torch.arange(0, 33, step=3, dtype=torch.long)[1:]

    # following is output onnx_decoder model code
    dummy_input = inputs[0]
    torch.onnx.export(rescore_model, (hyps_pad, hyps_lens, dummy_input),
                      rescore_model_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['hyps_pad', 'hyps_lens', 'encoder_out'],
                      output_names=['o1', 'o2'],
                      dynamic_axes={
                          'hyps_pad': {
                              0: 'batch_size',
                              1: 'subsample_len'
                          },
                          'hyps_lens': {
                              0: 'batch_size'
                          },
                          'encoder_out': {
                              1: 'batch_size',
                              2: 'hyp_max_len'
                          },
                          'o1': {
                              0: 'batch_size',
                              1: 'hyp_max_len'
                          },
                          'o2': {
                              0: 'batch_size',
                              1: 'hyp_max_len'
                          }}, verbose=True)
    print('export rescore model done')


def check_rescore_onnx_and_pytorch(model, torch_model, decoder_model_path):
    # following is test torch decoder's function forward code
    inputs = [torch.randn(1, 60 * (i + 1), output_size) for i in range(5)]
    hyps_pad = (abs(torch.randn(10, 30)) * 1000).ceil().long()
    hyps_lens = torch.arange(0, 33, step=3, dtype=torch.long)[1:]

    chunk_onnx_outputs = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        rescore_out, _ = model(hyps_pad, hyps_lens, dummy_input)
        chunk_onnx_outputs.append(rescore_out)

    torch_outputs = []
    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        rescore_out = model(hyps_pad, hyps_lens, dummy_input)
        torch_outputs.append(rescore_out)

    onnx_model = onnx.load(decoder_model_path)
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print("rescore onnx_model check pass!")

    ort_session = onnxruntime.InferenceSession(decoder_model_path)
    print(decoder_model_path + " onnx model has " +
          str(len(ort_session.get_inputs())) + " args")

    for i in range(5):
        # compute ONNX Runtime output prediction
        dummy_input = inputs[i]
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(hyps_pad),
            ort_session.get_inputs()[1].name: to_numpy(hyps_lens),
            ort_session.get_inputs()[2].name: to_numpy(dummy_input),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(torch_outputs[i][0]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   ort_outs[0],
                                   rtol=1e-03,
                                   atol=1e-05)
        np.testing.assert_allclose(to_numpy(chunk_onnx_outputs[i]),
                                   to_numpy(torch_outputs[i][0]),
                                   rtol=1e-03,
                                   atol=1e-05)

    print("Exported rescore model has been tested with ONNXRuntime, \
and the result looks good!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', default='train.yaml', help='config file')
    parser.add_argument('--checkpoint',
                        default='avg_8.pt',
                        help='checkpoint model')
    parser.add_argument('--output_dir',
                        default='onnx_model',
                        help='onnx output dir')
    parser.add_argument('--check_onnx', default=True, help='check onnx model')

    args = parser.parse_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    output_size = configs['encoder_conf']['output_size']
    num_blocks = configs['encoder_conf']['num_blocks']
    cnn_module_kernel = configs['encoder_conf']['cnn_module_kernel']

    model = init_asr_model(configs, onnx_mode=True)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    conf_path = os.path.join(output_dir, 'onnx.conf')
    export_conf(model, conf_path)

    encoder_model = model.encoder
    encoder_model.forward = encoder_model.forward_chunk_onnx
    encoder_model_path = os.path.join(output_dir, 'encoder.onnx')
    output_encoder_conformer_onnx(encoder_model, encoder_model_path)

    ctc_model = model.ctc
    ctc_model.forward = ctc_model.log_softmax
    ctc_model_path = os.path.join(output_dir, 'ctc.onnx')
    output_ctc_onnx(ctc_model, ctc_model_path)

    model.forward = model.forward_attention_decoder
    rescore_model_path = os.path.join(output_dir, 'rescore.onnx')
    output_rescore_onnx(model, rescore_model_path)

    if args.check_onnx:
        torch_model = init_asr_model(configs, onnx_mode=False)
        load_checkpoint(torch_model, args.checkpoint)
        torch_model.eval()
        torch_model.encoder.forward = torch_model.encoder.forward_chunk
        torch_model.ctc.forward = torch_model.ctc.log_softmax
        torch_model.forward = torch_model.forward_attention_decoder
        check_encoder_onnx_and_pytorch(encoder_model, torch_model.encoder,
                                       encoder_model_path)
        check_ctc_onnx_and_pytorch(ctc_model, torch_model.ctc, ctc_model_path)
        check_rescore_onnx_and_pytorch(model, torch_model, rescore_model_path)
