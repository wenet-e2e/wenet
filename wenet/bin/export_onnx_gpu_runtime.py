# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import sys

import torch
import yaml
import logging

from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.init_model import init_model

try:
    import onnxruntime
except ImportError:
    print('Please install onnxruntime-gpu!')
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class Encoder(torch.nn.Module):
    def __init__(self,
                 encoder: BaseEncoder,
                 ctc: CTC,
                 beam_size: int = 10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(self, speech: torch.Tensor,
                speech_lengths: torch.Tensor,):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        encoder_out, encoder_mask = self.encoder(speech,
                                                 speech_lengths,
                                                 -1, -1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        return encoder_out, encoder_out_lens, ctc_log_probs


class Decoder(torch.nn.Module):
    def __init__(self,
                 decoder: TransformerDecoder,
                 ctc_weight: float = 0.5,
                 reverse_weight: float = 0.0,
                 beam_size: int = 10,
                 eos: int = 5537):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size
        self.eos = eos

    def forward(self,
                encoder_out: torch.Tensor,
                hyps_pad_sos: torch.Tensor,
                hyps_lens_sos: torch.Tensor):
        """ Export interface for c++ call, forward decoder with batch of
            hypothesis from ctc prefix beam search and encoder output
        Args:
            encoder_out: B x T x F
            hyps_pad_sos: B x beam x T2
                        hyps with sos and padded by 0
            hyps_lens_sos: B x beam, length for each hyp with sos

        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
        """
        B, T, F = encoder_out.shape
        bz = hyps_pad_sos.shape[1]
        B2 = B * bz
        T2 = hyps_pad_sos.shape[2]
        # 1. prepare inputs for decoder
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = torch.ones(B2, 1, T,
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        hyps = hyps_pad_sos.view(B2, T2)
        hyps_lens = hyps_lens_sos.view(B2,)
        if self.reverse_weight > 0:
            r_hyps_lens = hyps_lens - 1
            r_hyps = hyps[:, 1:]
            max_len = torch.max(r_hyps_lens)
            index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
            seq_len_expand = r_hyps_lens.unsqueeze(1)
            seq_mask = seq_len_expand > index_range  # (beam, max_len)
            index = (seq_len_expand - 1) - index_range  # (beam, max_len)
            index = index * seq_mask
            r_hyps = torch.gather(r_hyps, 1, index)
            r_hyps = torch.where(seq_mask, r_hyps, self.eos)
            r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        else:
            r_hyps = torch.empty(0, device=encoder_out.device)

        # 2. decoding
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        r_decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        # V = decoder_out.shape[-1]
        # decoder_out = decoder_out.view(B, bz, T2, V)
        # print("decoder_out.shape", decoder_out.shape)
        # r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        return decoder_out, r_decoder_out  # B2 X T2 X V


def to_numpy(tensors):
    out = []
    if type(tensors) == torch.tensor:
        tensors = [tensors]
    for tensor in tensors:
        if tensor.requires_grad:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.cpu().numpy()
        out.append(tensor)
    return out


def test(xlist, blist, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=True):
    for a, b in zip(xlist, blist):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print(error)
            else:
                raise


def export_offline_encoder(model, configs, args, logger, encoder_onnx_path):
    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(
            low=10, high=seq_len, size=(bz,), dtype=torch.int32)
    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()

    torch.onnx.export(encoder,
                      (speech, speech_lens),
                      encoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['speech', 'speech_lengths'],
                      output_names=['encoder_out', 'encoder_out_lens',
                                    'ctc_log_probs'],
                      dynamic_axes={
                          'speech': {0: 'B', 1: 'T'},
                          'speech_lengths': {0: 'B'},
                          'encoder_out': {0: 'B', 1: 'T_OUT'},
                          'encoder_out_lens': {0: 'B'},
                          'ctc_log_probs': {0: 'B', 1: 'T_OUT'},
                      },
                      verbose=False
                      )

    with torch.no_grad():
        o0, o1, o2 = encoder(speech, speech_lens)

    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path,
                                               providers=providers)
    ort_inputs = {'speech': to_numpy(speech),
                  'speech_lengths': to_numpy(speech_lens)}
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output
    test(to_numpy([o0, o1, o2]), ort_outs)
    logger.info("export offline onnx encoder succeed!")
    is_bidirectional_decoder = 1 if configs['decoder'] == 'bitransformer' else 0
    onnx_config = {"beam_size": args.beam_size,
                   "reverse_weight": args.reverse_weight,
                   "ctc_weight": args.ctc_weight,
                   "sos": configs["output_dim"] - 1,
                   "eos": configs["output_dim"] - 1,
                   "is_bidirectional_decoder": is_bidirectional_decoder,
                   "fp16": args.fp16}
    return onnx_config


def export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path):
    bz, seq_len = 32, 100
    beam_size = args.beam_size
    decoder = Decoder(model.decoder,
                      model.ctc_weight,
                      model.reverse_weight,
                      beam_size,
                      configs["output_dim"] - 1)
    decoder.eval()

    hyps_pad_sos_eos = torch.randint(
            low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(
            low=3, high=seq_len, size=(bz, beam_size), dtype=torch.int32)

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)

    input_names = ['encoder_out', 'hyps_pad_sos', 'hyps_lens_sos', ]

    torch.onnx.export(decoder,
                      (encoder_out,
                       hyps_pad_sos_eos, hyps_lens_sos),
                      decoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=['decoder_out', 'r_decoder_out'],
                      dynamic_axes={'encoder_out': {0: 'B', 1: 'T'},
                                    'hyps_pad_sos': {0: 'B', 2: 'T2'},
                                    'hyps_lens_sos': {0: 'B'},
                                    'decoder_out': {0: 'B'},
                                    'r_decoder_out': {0: 'B'},
                                    },
                      verbose=False
                      )
    with torch.no_grad():
        o0 = decoder(encoder_out,
                     hyps_pad_sos_eos,
                     hyps_lens_sos,)
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(decoder_onnx_path,
                                               providers=providers)

    input_tensors = [encoder_out, hyps_pad_sos_eos,
                     hyps_lens_sos]
    ort_inputs = {}
    input_tensors = to_numpy(input_tensors)
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]

    # if model.reverse weight == 0,
    # the r_hyps_pad will be removed
    # from the onnx decoder since it doen't play any role
    # if model.reverse_weight == 0:
    #     del ort_inputs['r_hyps_pad_sos_eos']
    ort_outs = ort_session.run(None, ort_inputs)

    # check decoder output
    test(to_numpy(list(o0)), ort_outs, rtol=1e-03, atol=1e-05)
    logger.info("export to onnx decoder succeed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export x86_gpu model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn_file', required=False, default='', type=str,
                        help='global_cmvn file, default path is in config file')
    parser.add_argument('--reverse_weight', default=-1.0, type=float,
                        required=False,
                        help='reverse weight for bitransformer,' +
                        'default value is in config file')
    parser.add_argument('--ctc_weight', default=-1.0, type=float,
                        required=False,
                        help='ctc weight, default value is in config file')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_onnx_dir',
                        default="onnx_model",
                        help='output onnx encoder and decoder directory')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if args.cmvn_file and os.path.exists(args.cmvn_file):
        configs['cmvn_file'] = args.cmvn_file
    if args.reverse_weight != -1.0 and 'reverse_weight' in configs['model_conf']:
        configs['model_conf']['reverse_weight'] = args.reverse_weight
        print("Update reverse weight to", args.reverse_weight)
    if args.ctc_weight != -1:
        print("Update ctc weight to ", args.ctc_weight)
        configs['model_conf']['ctc_weight'] = args.ctc_weight
    configs["encoder_conf"]["use_dynamic_chunk"] = False

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    if not os.path.exists(args.output_onnx_dir):
        os.mkdir(args.output_onnx_dir)
    encoder_onnx_path = os.path.join(args.output_onnx_dir, 'encoder.onnx')
    export_enc_func = export_offline_encoder

    onnx_config = export_enc_func(model, configs, args, logger, encoder_onnx_path)

    decoder_onnx_path = os.path.join(args.output_onnx_dir, 'decoder.onnx')
    export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path)

    if args.fp16:
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import convert_float_to_float16
        except ImportError:
            import traceback
            traceback.print_exc()
            print('Please install onnxmltools!')
            sys.exit(1)
        encoder_onnx_model = onnxmltools.utils.load_model(encoder_onnx_path)
        encoder_onnx_model = convert_float_to_float16(encoder_onnx_model)
        encoder_onnx_path = os.path.join(args.output_onnx_dir, 'encoder_fp16.onnx')
        onnxmltools.utils.save_model(encoder_onnx_model, encoder_onnx_path)
        decoder_onnx_model = onnxmltools.utils.load_model(decoder_onnx_path)
        decoder_onnx_model = convert_float_to_float16(decoder_onnx_model)
        decoder_onnx_path = os.path.join(args.output_onnx_dir, 'decoder_fp16.onnx')
        onnxmltools.utils.save_model(decoder_onnx_model, decoder_onnx_path)
    # dump configurations

    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
