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


def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    encoder = asr_model.encoder
    encoder.forward = encoder.forward_chunk
    encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')

    print("\tStage-1.1: prepare inputs for encoder")
    chunk = torch.randn(
        (args['batch'], args['decoding_window'], args['feature_size']))
    offset = 0
    # NOTE(xcsong): The uncertainty of `next_cache_start` only appears
    #   in the first few chunks, this is caused by dynamic att_cache shape, i,e
    #   (0, 0, 0, 0) for 1st chunk and (elayers, head, ?, d_k*2) for subsequent
    #   chunks. One way to ease the ONNX export is to keep `next_cache_start`
    #   as a fixed value. To do this, for the **first** chunk, if
    #   left_chunks > 0, we feed real cache & real mask to the model, otherwise
    #   fake cache & fake mask. In this way, we get:
    #   1. 16/-1 mode: next_cache_start == 0 for all chunks
    #   2. 16/4  mode: next_cache_start == chunk_size for all chunks
    #   3. 16/0  mode: next_cache_start == chunk_size for all chunks
    #   4. -1/-1 mode: next_cache_start == 0 for all chunks
    #   NO MORE DYNAMIC CHANGES!!
    if args['left_chunks'] > 0:  # 16/4
        required_cache_size = args['chunk_size'] * args['left_chunks']
        offset = required_cache_size
        # Real cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], required_cache_size,
             args['output_size'] // args['head'] * 2))
        # Real mask
        att_mask = torch.ones(
            (args['batch'], 1, required_cache_size + args['chunk_size']),
            dtype=torch.bool)
        att_mask[:, :, :required_cache_size] = 0
    elif args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
        required_cache_size = -1 if args['left_chunks'] < 0 else 0
        # Fake cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], 0,
             args['output_size'] // args['head'] * 2))
        # Fake mask
        att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    cnn_cache = torch.zeros(
        (args['num_blocks'], args['batch'],
         args['output_size'], args['cnn_module_kernel'] - 1))
    inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset),
          "\t\trequired_cache: {}\n".format(required_cache_size),
          "\t\tatt_cache.size(): {}\n".format(att_cache.size()),
          "\t\tcnn_cache.size(): {}\n".format(cnn_cache.size()),
          "\t\tatt_mask.size(): {}\n".format(att_mask.size()))

    print("\tStage-1.2: torch.onnx.export")
    dynamic_axes = {
        'chunk': {1: 'T'},
        'att_cache': {2: 'T_CACHE'},
        'output': {1: 'T'},
        'r_att_cache': {2: 'T_CACHE'},
    }
    if args['chunk_size'] > 0:  # 16/4, 16/-1, 16/0
        dynamic_axes.pop('chunk')
        dynamic_axes.pop('output')
    if args['left_chunks'] >= 0:  # 16/4, 16/0
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
    print("\t\tonnx_encoder inputs : {}".format(
        [node.name for node in onnx_encoder.graph.input]))
    print("\t\tonnx_encoder outputs: {}".format(
        [node.name for node in onnx_encoder.graph.output]))
    print('\t\tExport onnx_encoder, done! see {}'.format(
        encoder_outpath))

    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        print("\t\ttorch chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, list(torch_chunk.size()), torch_offset,
                  list(torch_att_cache.size()),
                  list(torch_cnn_cache.size()), list(torch_att_mask.size())))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            torch_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
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
        print("\t\tonnx  chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, onnx_chunk.shape, onnx_offset, onnx_att_cache.shape,
                  onnx_cnn_cache.shape, onnx_att_mask.shape))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            onnx_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
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
        if args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
            ort_inputs.pop('required_cache_size')
            ort_inputs.pop('att_mask')
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
        onnx_offset += ort_outs[0].shape[1]
    onnx_output = np.concatenate(onnx_output, axis=1)

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_encoder, pass!")


def export_ctc(asr_model, args):
    print("Stage-2: export ctc")
    ctc = asr_model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(args['output_dir'], 'ctc.onnx')

    print("\tStage-2.1: prepare inputs for ctc")
    hidden = torch.randn(
        (args['batch'], args['chunk_size'] if args['chunk_size'] > 0 else 16,
         args['output_size']))

    print("\tStage-2.2: torch.onnx.export")
    dynamic_axes = {'hidden': {1: 'T'}, 'probs': {1: 'T'}}
    if args['chunk_size'] > 0:  # 16/4, 16/-1, 16/0
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
    print("\t\tonnx_ctc inputs : {}".format(
        [node.name for node in onnx_ctc.graph.input]))
    print("\t\tonnx_ctc outputs: {}".format(
        [node.name for node in onnx_ctc.graph.output]))
    print('\t\tExport onnx_ctc, done! see {}'.format(
        ctc_outpath))

    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {'hidden' : to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_ctc, pass!")


def export_decoder(asr_model, args):
    print("Stage-3: export decoder")
    decoder = asr_model.decoder
    decoder_outpath = os.path.join(args['output_dir'], 'decoder.onnx')

    print("\tStage-3.1: prepare inputs for decoder")
    # hardcode time->200 len->20, they are dynamic axes.
    encoder_out = torch.randn((args['beam'], 200, args['output_size']))
    encoder_mask = torch.zeros((0, 0, 0))  # fake mask, never used.
    hyps = torch.randint(low=0, high=args['vocab_size'],
                         size=[args['beam'], 20])
    hyps[:, 0] = args['vocab_size'] - 1  # <sos>
    hyps_lens = torch.randint(low=15, high=21, size=[args['beam']])
    r_hyps = reverse_pad_list(
        hyps[:, 1:], hyps_lens - 1, float(asr_model.ignore_id))
    r_hyps, _ = add_sos_eos(r_hyps, asr_model.sos, asr_model.eos,
                            asr_model.ignore_id)

    print("\tStage-3.2: torch.onnx.export")
    dynamic_axes = {
        'encoder_out': {1: 'T'}, 'hyps': {1: 'L'}, 'r_hyps': {1: 'L'},
        'logits': {1: 'L'}, 'r_logits': {1: 'L'}
    }
    inputs = (encoder_out, encoder_mask, hyps,
              hyps_lens, r_hyps, args['reverse_weight'])
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
    print("\t\tonnx_decoder inputs : {}".format(
        [node.name for node in onnx_decoder.graph.input]))
    print("\t\tonnx_decoder outputs: {}".format(
        [node.name for node in onnx_decoder.graph.output]))
    print('\t\tExport onnx_decoder, done! see {}'.format(
        decoder_outpath))

    print("\tStage-3.3: check onnx_decoder and torch_decoder")
    torch_logits, torch_r_logits, _ = decoder(
        encoder_out, encoder_mask, hyps, hyps_lens,
        r_hyps, args['reverse_weight'])
    ort_session = onnxruntime.InferenceSession(decoder_outpath)
    ort_inputs = {
        'encoder_out': to_numpy(encoder_out),
        'hyps': to_numpy(hyps),
        'hyps_lens': to_numpy(hyps_lens),
        'r_hyps': to_numpy(r_hyps),
    }
    if args['reverse_weight'] == 0.0:  # r_hyps is removed by ONNX automatically
        ort_inputs.pop('r_hyps')
    onnx_output = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_logits), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    if args['reverse_weight'] > 0.0:
        np.testing.assert_allclose(to_numpy(torch_r_logits), onnx_output[1],
                                   rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_decoder, pass!")


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

    arguments = {}
    arguments['output_dir'] = output_dir
    arguments['batch'] = 1
    arguments['chunk_size'] = args.chunk_size
    arguments['left_chunks'] = args.num_decoding_left_chunks
    arguments['beam'] = args.beam
    arguments['reverse_weight'] = args.reverse_weight
    arguments['output_size'] = configs['encoder_conf']['output_size']
    arguments['num_blocks'] = configs['encoder_conf']['num_blocks']
    arguments['cnn_module_kernel'] = configs['encoder_conf']['cnn_module_kernel']
    arguments['head'] = configs['encoder_conf']['attention_heads']
    arguments['feature_size'] = configs['input_dim']
    arguments['vocab_size'] = configs['output_dim']
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    arguments['decoding_window'] = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1 if args.chunk_size > 0 else 67

    export_encoder(model, arguments)
    export_ctc(model, arguments)
    export_decoder(model, arguments)

if __name__ == '__main__':
    main()
