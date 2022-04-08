#!/usr/bin/env python3
# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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

from __future__ import print_function

import argparse
import os
import copy
import sys

import torch
import yaml
import numpy as np

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.common import reverse_pad_list, add_sos_eos

try:
    import onnx
    import onnxruntime
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--chunk_size', required=True,
                        type=int, help='decoding chunk size')
    parser.add_argument('--num_decoding_left_chunks', required=True,
                        type=int, help='cache chunks')
    parser.add_argument('--beam', required=True,
                        type=int, help='beam wigth')
    parser.add_argument('--reverse_weight', default=0.0,
                        type=float, help='reverse_weight in attention_rescoing')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)

    batch = 1
    chunk_size = args.chunk_size
    left_chunks = args.num_decoding_left_chunks
    beam = args.beam
    reverse_weight = args.reverse_weight
    output_size = configs['encoder_conf']['output_size']
    num_blocks = configs['encoder_conf']['num_blocks']
    cnn_module_kernel = configs['encoder_conf']['cnn_module_kernel']
    head = configs['encoder_conf']['attention_heads']
    feature_size = configs['input_dim']
    vocab_size = configs['output_dim']
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    decoding_window = (chunk_size - 1) * model.encoder.embed.subsampling_rate \
        + model.encoder.embed.right_context + 1 if chunk_size > 0 else 67

    # ============================= encoder =================================
    print("\033[32mStage-1: export encoder\033[0m")
    encoder = model.encoder
    encoder.forward = encoder.forward_chunk
    encoder_outpath = os.path.join(output_dir, 'encoder.onnx')

    print("\t\033[32mStage-1.1: prepare inputs for encoder\033[0m")
    chunk = torch.randn((batch, decoding_window, feature_size))
    offset = 0
    # NOTE(xcsong): if left_chunks > 0, we feed real cache & real mask
    #   to the model, otherwise fake cache & fake mask. This is to ensure
    #   that all if-else branches in `forward_chunk` will always take the
    #   same path with a given setup(i.e., 16/-1, 16/4, 16/0 or -1/-1)
    if left_chunks > 0:  # 16/4
        required_cache_size = chunk_size * left_chunks
        offset = required_cache_size
        # Real cache
        att_cache = torch.zeros(
            (num_blocks, head, required_cache_size, output_size // head * 2))
        # Real mask
        att_mask = torch.ones(
            (batch, 1, required_cache_size + chunk_size), dtype=torch.bool)
        att_mask[:, :, :required_cache_size] = 0
    elif left_chunks <= 0:  # 16/-1, -1/-1, 16/0
        required_cache_size = -1 if left_chunks < 0 else 0
        # Fake cache
        att_cache = torch.zeros(
            (num_blocks, head, 0, output_size // head * 2))
        # Fake mask
        att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    cnn_cache = torch.zeros(
        (num_blocks, batch, output_size, cnn_module_kernel - 1))
    inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
    print("\t\t\033[32mchunk.size(): {}\033[0m\n".format(chunk.size()),
          "\t\t\033[32moffset: {}\033[0m\n".format(offset),
          "\t\t\033[32mrequired_cache: {}\033[0m\n".format(required_cache_size),
          "\t\t\033[32matt_cache.size(): {}\033[0m\n".format(att_cache.size()),
          "\t\t\033[32mcnn_cache.size(): {}\033[0m\n".format(cnn_cache.size()),
          "\t\t\033[32matt_mask.size(): {}\033[0m\n".format(att_mask.size()))

    print("\t\033[32mStage-1.2: torch.onnx.export\033[0m")
    dynamic_axes = {
        'chunk': {1: 'T'},
        'att_cache': {2: 'T_CACHE'},
        'output': {1: 'T'},
        'r_att_cache': {2: 'T_CACHE'},
    }
    if chunk_size > 0:  # 16/4, 16/-1, 16/0
        dynamic_axes.pop('chunk')
        dynamic_axes.pop('output')
    if left_chunks >= 0:  # 16/4, 16/0
        # NOTE(xsong): since we feed real cache & real mask into the
        #   model when left_chunks > 0, the shape of cache will never
        #   be changed.
        dynamic_axes.pop('att_cache')
        dynamic_axes.pop('r_att_cache')
    torch.onnx.export(
        encoder, inputs, encoder_outpath, opset_version=14,
        export_params=True, do_constant_folding=True,
        input_names=[
            'chunk', 'offset', 'required_cache_size',
            'att_cache', 'cnn_cache', 'att_mask'
        ],
        output_names=['output', 'r_att_cache', 'r_cnn_cache'],
        dynamic_axes=dynamic_axes, verbose=False)
    onnx_encoder = onnx.load(encoder_outpath)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    print("\t\t\033[32monnx_encoder inputs : {}\033[0m".format(
        [node.name for node in onnx_encoder.graph.input]))
    print("\t\t\033[32monnx_encoder outputs: {}\033[0m".format(
        [node.name for node in onnx_encoder.graph.output]))
    print('\t\t\033[32mExport onnx_encoder, done! see {}\033[0m'.format(
        encoder_outpath))

    print("\t\033[32mStage-1.3: check onnx_encoder and torch_encoder\033[0m")
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if left_chunks > 0:  # 16/4
            torch_att_mask[:, :, -(chunk_size * (i + 1)):] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_offset, torch_required_cache_size,
            torch_att_cache, torch_cnn_cache, torch_att_mask)
        torch_output.append(out)
        torch_offset += out.size(1)
    torch_output = torch.cat(torch_output, dim=1)

    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = np.array((offset)).astype(np.int64)
    onnx_required_cache_size = np.array((required_cache_size)).astype(np.int64)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    for i in range(10):
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if left_chunks > 0:  # 16/4
            onnx_att_mask[:, :, -(chunk_size * (i + 1)):] = 1
        ort_inputs = {
            'chunk': onnx_chunk, 'offset': onnx_offset,
            'required_cache_size': onnx_required_cache_size,
            'att_cache': onnx_att_cache, 'cnn_cache': onnx_cnn_cache,
            'att_mask': onnx_att_mask
        }
        # NOTE(xcsong): If we use 16/-1, -1/-1 or 16/0 mode, `next_cache_start`
        #   will be hardcoded to 0 or chunk_size by ONNX, thus
        #   required_cache_size and att_mask are no more needed and they will
        #   be removed by ONNX automatically.
        if left_chunks <= 0:  # 16/-1, -1/-1, 16/0
            ort_inputs.pop('required_cache_size')
            ort_inputs.pop('att_mask')
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
        onnx_offset += ort_outs[0].shape[1]
    onnx_output = np.concatenate(onnx_output, axis=1)

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    print("\t\t\033[32mCheck onnx_encoder, pass!\033[0m")

    # ============================= ctc =================================
    print("\033[32mStage-2: export ctc\033[0m")
    ctc = model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(output_dir, 'ctc.onnx')

    print("\t\033[32mStage-2.1: prepare inputs for ctc\033[0m")
    hidden = torch.randn(
        (batch, chunk_size if chunk_size > 0 else 16, output_size))

    print("\t\033[32mStage-2.2: torch.onnx.export\033[0m")
    dynamic_axes = {'hidden': {1: 'T'}, 'probs': {1: 'T'}}
    if chunk_size > 0:  # 16/4, 16/-1, 16/0
        dynamic_axes.pop('hidden')
        dynamic_axes.pop('probs')
    torch.onnx.export(
        ctc, hidden, ctc_outpath, opset_version=14,
        export_params=True, do_constant_folding=True,
        input_names=['hidden'], output_names=['probs'],
        dynamic_axes=dynamic_axes, verbose=False)
    onnx_ctc = onnx.load(ctc_outpath)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    print("\t\t\033[32monnx_ctc inputs : {}\033[0m".format(
        [node.name for node in onnx_ctc.graph.input]))
    print("\t\t\033[32monnx_ctc outputs: {}\033[0m".format(
        [node.name for node in onnx_ctc.graph.output]))
    print('\t\t\033[32mExport onnx_ctc, done! see {}\033[0m'.format(
        ctc_outpath))

    print("\t\033[32mStage-2.3: check onnx_ctc and torch_ctc\033[0m")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {'hidden' : to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\t\t\033[32mCheck onnx_ctc, pass!\033[0m")

    # ============================= decoder =================================
    print("\033[32mStage-3: export decoder\033[0m")
    decoder = model.decoder
    decoder_outpath = os.path.join(output_dir, 'decoder.onnx')

    print("\t\033[32mStage-3.1: prepare inputs for decoder\033[0m")
    # hardcode time->200 len->20, they are dynamic axes.
    encoder_out = torch.randn((beam, 200, output_size))
    encoder_mask = torch.zeros((0, 0, 0))  # fake mask, never used.
    hyps = torch.randint(low=0, high=vocab_size, size=[beam, 20])
    hyps[:, 0] = vocab_size - 1  # <sos>
    hyps_lens = torch.randint(low=15, high=21, size=[beam])
    r_hyps = reverse_pad_list(
        hyps[:, 1:], hyps_lens - 1, float(model.ignore_id))
    r_hyps, _ = add_sos_eos(r_hyps, model.sos, model.eos, model.ignore_id)

    print("\t\033[32mStage-3.2: torch.onnx.export\033[0m")
    dynamic_axes = {
        'encoder_out': {1: 'T'}, 'hyps': {1: 'L'}, 'r_hyps': {1: 'L'},
        'logits': {1: 'L'}, 'r_logits': {1: 'L'}
    }
    inputs = (encoder_out, encoder_mask, hyps,
              hyps_lens, r_hyps, reverse_weight)
    torch.onnx.export(
        decoder, inputs, decoder_outpath, opset_version=14,
        export_params=True, do_constant_folding=True,
        input_names=[
            'encoder_out', 'encoder_mask', 'hyps', 'hyps_lens', 'r_hyps'],
        output_names=[
            'logits_before_logsoftmax', 'r_logits_before_logsoftmax', 'olens'],
        dynamic_axes=dynamic_axes, verbose=False)
    onnx_decoder = onnx.load(decoder_outpath)
    onnx.checker.check_model(onnx_decoder)
    onnx.helper.printable_graph(onnx_decoder.graph)
    print("\t\t\033[32monnx_decoder inputs : {}\033[0m".format(
        [node.name for node in onnx_decoder.graph.input]))
    print("\t\t\033[32monnx_decoder outputs: {}\033[0m".format(
        [node.name for node in onnx_decoder.graph.output]))
    print('\t\t\033[32mExport onnx_decoder, done! see {}\033[0m'.format(
        decoder_outpath))

    print("\t\033[32mStage-3.3: check onnx_decoder and torch_decoder\033[0m")
    torch_logits, torch_r_logits, _ = decoder(
        encoder_out, encoder_mask, hyps, hyps_lens, r_hyps, reverse_weight)
    ort_session = onnxruntime.InferenceSession(decoder_outpath)
    ort_inputs = {
        'encoder_out': to_numpy(encoder_out),
        'hyps': to_numpy(hyps),
        'hyps_lens': to_numpy(hyps_lens),
        'r_hyps': to_numpy(r_hyps),
    }
    if reverse_weight == 0.0:  # r_hyps is removed by ONNX automatically.
        ort_inputs.pop('r_hyps')
    onnx_output = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_logits), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    if reverse_weight > 0.0:
        np.testing.assert_allclose(to_numpy(torch_r_logits), onnx_output[1],
                                   rtol=1e-03, atol=1e-05)
    print("\t\t\033[32mCheck onnx_decoder, pass!\033[0m")

if __name__ == '__main__':
    main()
