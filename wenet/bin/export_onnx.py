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
                 ctc_weight: float = 0.5,
                 reverse_weight: float = 0.0,
                 beam_size: int = 10):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_lens: torch.Tensor,
                hyps_pad_sos_eos: torch.Tensor,
                hyps_lens_sos: torch.Tensor,
                r_hyps_pad_sos_eos: torch.Tensor,
                ctc_score: torch.Tensor):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            r_hyps_pad_sos_eos: B x beam x (T2+1),
                    reversed hyps with sos & eos and padded by ignore id
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(B2,)
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()

        r_hyps_pad = r_hyps_pad_sos_eos.view(B2, T2 + 1)
        r_hyps_pad_sos = r_hyps_pad[:, :-1].contiguous()
        r_hyps_pad_eos = r_hyps_pad[:, 1:].contiguous()

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad_sos, hyps_lens, r_hyps_pad_sos,
            self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)  # B2 x T2
        # mask index, remove ignore id
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)  # B2 X T2
        # mask padded part
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)
        if self.reverse_weight > 0:
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
            r_decoder_out = r_decoder_out.view(B2, T2, V)
            index = torch.unsqueeze(r_hyps_pad_eos * mask, 2)
            r_score = r_decoder_out.gather(2, index).squeeze(2)
            r_score = r_score * mask
            score = score * (1 - self.reverse_weight) + self.reverse_weight * r_score
            r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        score = torch.sum(score, axis=1)  # B2
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        return best_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
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

    def test(a, b, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=True):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print(error)
            else:
                raise

    # check encoder output
    test(to_numpy(o0), ort_outs[0], rtol=1e-03, atol=1e-5)
    test(to_numpy(o1), ort_outs[1], rtol=1e-03, atol=1e-05)
    test(to_numpy(o2), ort_outs[2], rtol=1e-03, atol=1e-05)
    test(to_numpy(o3), ort_outs[3], rtol=1e-03, atol=1e-05)
    test(to_numpy(o4), ort_outs[4], rtol=1e-03, atol=1e-05)
    logger.info("export to onnx encoder succeed!")

    decoder = Decoder(
        model.decoder,
        model.ctc_weight,
        model.reverse_weight,
        beam_size)
    decoder.eval()
    decoder_onnx_path = os.path.join(args.output_onnx_dir, 'decoder.onnx')

    hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(low=3, high=seq_len, size=(bz, beam_size),
                                  dtype=torch.int32)
    r_hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3, high=seq_len, size=(bz,), dtype=torch.int32)
    ctc_score = torch.randn(bz, beam_size, dtype=torch.float32)
    torch.onnx.export(decoder,
                      (encoder_out, encoder_out_lens,
                       hyps_pad_sos_eos, hyps_lens_sos,
                       r_hyps_pad_sos_eos, ctc_score),
                      decoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['encoder_out', 'encoder_out_lens',
                                   'hyps_pad_sos_eos', 'hyps_lens_sos',
                                   'r_hyps_pad_sos_eos', 'ctc_score'],
                      output_names=['best_index'],
                      dynamic_axes={'encoder_out': {0: 'B', 1: 'T'},
                                    'encoder_out_lens': {0: 'B'},
                                    'hyps_pad_sos_eos': {0: 'B', 2: 'T2'},
                                    'hyps_lens_sos': {0: 'B'},
                                    'r_hyps_pad_sos_eos': {0: 'B', 2: 'T2'},
                                    'ctc_score': {0: 'B'},
                                    'best_index': {0: 'B'},
                                    },
                      verbose=False
                      )
    with torch.no_grad():
        o0 = decoder(
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos_eos,
            hyps_lens_sos,
            r_hyps_pad_sos_eos,
            ctc_score)

    ort_session = onnxruntime.InferenceSession(decoder_onnx_path,
                                               providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(encoder_out),
                  ort_session.get_inputs()[1].name: to_numpy(encoder_out_lens),
                  ort_session.get_inputs()[2].name: to_numpy(hyps_pad_sos_eos),
                  ort_session.get_inputs()[3].name: to_numpy(hyps_lens_sos),
                  ort_session.get_inputs()[-1].name: to_numpy(ctc_score)
                  }
    # if model.reverse weight == 0,
    # the r_hyps_pad will be removed
    # from the onnx decoder since it doen't play any role
    if model.reverse_weight > 0:
        ort_inputs[ort_session.get_inputs()[4].name] = to_numpy(r_hyps_pad_sos_eos)
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output
    test(to_numpy(o0), ort_outs[0], rtol=1e-03, atol=1e-05)
    logger.info("export to onnx decoder succeed!")

    if args.fp16:
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import convert_float_to_float16
        except ImportError:
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
    onnx_config = {"beam_size": args.beam_size,
                   "reverse_weight": args.reverse_weight,
                   "ctc_weight": args.ctc_weight,
                   "fp16": args.fp16}

    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
