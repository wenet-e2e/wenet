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

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.mask import make_pad_mask

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
        beam_log_probs, beam_log_probs_idx = torch.topk(
            ctc_log_probs, self.beam_size, dim=2)
        return encoder_out, encoder_out_lens, ctc_log_probs, \
            beam_log_probs, beam_log_probs_idx


class Decoder(torch.nn.Module):
    def __init__(self,
                 decoder: TransformerDecoder,
                 reverse_weight: float = 0.0,
                 beam_size: int = 10):
        super().__init__()
        self.decoder = decoder
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_lens: torch.Tensor,
                hyps_pad_sos: torch.Tensor,
                hyps_lens: torch.Tensor,
                r_hyps_pad_sos: torch.Tensor):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos: B x beam x T2,
                        hyps with sos and padded by ignore id
            hyps_lens: B x beam, length for each hyp with sos
            r_hyps_pad_sos: B x beam x T2,
                    reversed hyps with sos and padded by ignore id
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos.shape[2]
        hyps_pad = hyps_pad_sos.view(B2, T2)
        hyps_lens = hyps_lens.view(B2,)
        r_hyps_pad = r_hyps_pad_sos.view(B2, T2)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B, bz, T2, V)
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        return decoder_out, r_decoder_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_onnx_dir',
                        default="onnx_model",
                        help='output onnx encoder and decoder directory')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    configs["encoder_conf"]["use_dynamic_chunk"] = False
    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(low=10, high=seq_len, size=(bz,), dtype=torch.int32)
    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()
    if not os.path.exists(args.output_onnx_dir):
        os.mkdir(args.output_onnx_dir)
    encoder_onnx_path = os.path.join(args.output_onnx_dir, 'encoder.onnx')

    torch.onnx.export(encoder,
                      (speech, speech_lens),
                      encoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['speech', 'speech_lengths'],
                      output_names=['encoder_out', 'encoder_out_lens',
                                    'ctc_log_probs',
                                    'beam_log_probs', 'beam_log_probs_idx'],
                      dynamic_axes={
                          'speech': {0: 'B', 1: 'T'},
                          'speech_lengths': {0: 'B'},
                          'encoder_out': {0: 'B', 1: 'T_OUT'},
                          'encoder_out_lens': {0: 'B'},
                          'ctc_log_probs': {0: 'B', 1: 'T_OUT'},
                          'beam_log_probs': {0: 'B', 1: 'T_OUT'},
                          'beam_log_probs_idx': {0: 'B', 1: 'T_OUT'},
                      },
                      verbose=False
                      )

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    with torch.no_grad():
        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)

    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path,
                                               providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(speech),
                  ort_session.get_inputs()[1].name: to_numpy(speech_lens)}
    ort_outs = ort_session.run(None, ort_inputs)

    def test(a, b, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=False):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print(error)
            else:
                raise

    # check encoder output
    test(to_numpy(o0), ort_outs[0], rtol=1e-03, atol=1e-05)
    test(to_numpy(o1), ort_outs[1], rtol=1e-03, atol=1e-05)
    test(to_numpy(o2), ort_outs[2], rtol=1e-03, atol=1e-05)
    test(to_numpy(o3), ort_outs[3], rtol=1e-03, atol=1e-05)
    test(to_numpy(o4), ort_outs[4], rtol=1e-03, atol=1e-05)
    logger.info("export to onnx encoder succeed!")

    decoder = Decoder(
        model.decoder,
        model.reverse_weight,
        beam_size)
    decoder.eval()
    decoder_onnx_path = os.path.join(args.output_onnx_dir, 'decoder.onnx')

    hyps_pad_sos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens = torch.randint(low=3, high=seq_len, size=(bz, beam_size),
                              dtype=torch.int32)
    r_hyps_pad_sos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3, high=seq_len, size=(bz,), dtype=torch.int32)

    torch.onnx.export(decoder,
                      (encoder_out, encoder_out_lens,
                       hyps_pad_sos, hyps_lens, r_hyps_pad_sos),
                      decoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['encoder_out', 'encoder_out_lens',
                                   'hyps_pad_sos', 'hyps_lens',
                                   'r_hyps_pad_sos'],
                      output_names=['decoder_out', 'r_decoder_out'],
                      dynamic_axes={'encoder_out': {0: 'B', 1: 'T'},
                                    'encoder_out_lens': {0: 'B'},
                                    'hyps_pad_sos': {0: 'B', 2: 'T2'},
                                    'hyps_lens': {0: 'B'},
                                    'r_hyps_pad_sos': {0: 'B', 2: 'T2'},
                                    'decoder_out': {0: 'B', 2: 'T2'},
                                    'r_decoder_out': {0: 'B', 2: 'T2'}
                                    },
                      verbose=False
                      )
    with torch.no_grad():
        o0, o1 = decoder(
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos,
            hyps_lens,
            r_hyps_pad_sos)

    ort_session = onnxruntime.InferenceSession(decoder_onnx_path,
                                               providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(encoder_out),
                  ort_session.get_inputs()[1].name: to_numpy(encoder_out_lens),
                  ort_session.get_inputs()[2].name: to_numpy(hyps_pad_sos),
                  ort_session.get_inputs()[3].name: to_numpy(hyps_lens),
                  ort_session.get_inputs()[4].name: to_numpy(r_hyps_pad_sos)
                  }
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output
    test(to_numpy(o0), ort_outs[0], rtol=1e-03, atol=1e-05)
    test(to_numpy(o1), ort_outs[1], rtol=1e-03, atol=1e-05)
    logger.info("export to onnx decoder succeed!")
