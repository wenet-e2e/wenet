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


class StreamingEncoder(torch.nn.Module):
    def __init__(self, model, required_cache_size, beam_size, transformer=False):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.transformer = transformer

    def forward(self, chunk_xs, chunk_lens, offset,
                att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        # B X 1 X T
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk
        # <---------forward_chunk START--------->
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)  # required cache size
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoder.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, masks, pos_emb,
                att_cache=att_cache[i],
                cnn_cache=cnn_cache[i])
            #   shape(new_att_cache) is (B, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (B, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :].unsqueeze(1))
            if not self.transformer:
                r_cnn_cache.append(new_cnn_cache.unsqueeze(1))
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)

        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx
        if not self.transformer:
            r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers

        # <---------forward_chunk END--------->

        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs,
                                              self.beam_size,
                                              dim=2)
        log_probs = log_probs.to(chunk_xs.dtype)

        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        chunk_out_lens = chunk_lens // self.subsampling_rate
        r_offset = r_offset.unsqueeze(1)

        return log_probs, log_probs_idx, chunk_out, chunk_out_lens, \
            r_offset, r_att_cache, r_cnn_cache, r_cache_mask


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
    speech_lens = torch.randint(low=10, high=seq_len, size=(bz,), dtype=torch.int32)
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

    with torch.no_grad():
        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)

    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path,
                                               providers=providers)
    ort_inputs = {'speech': to_numpy(speech),
                  'speech_lengths': to_numpy(speech_lens)}
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output
    test(to_numpy([o0, o1, o2, o3, o4]), ort_outs)
    logger.info("export offline onnx encoder succeed!")
    onnx_config = {"beam_size": args.beam_size,
                   "reverse_weight": args.reverse_weight,
                   "ctc_weight": args.ctc_weight,
                   "fp16": args.fp16}
    return onnx_config

def export_online_encoder(model, configs, args, logger, encoder_onnx_path):
    decoding_chunk_size = args.decoding_chunk_size
    subsampling = model.encoder.embed.subsampling_rate
    context = model.encoder.embed.right_context + 1
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    batch_size = 32
    audio_len = decoding_window
    feature_size = configs["input_dim"]
    output_size = configs["encoder_conf"]["output_size"]
    num_layers = configs["encoder_conf"]["num_blocks"]
    # in transformer the cnn module will not be available
    transformer = False
    cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
    if not cnn_module_kernel:
        transformer = True
    num_decoding_left_chunks = args.num_decoding_left_chunks
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    encoder = StreamingEncoder(model, required_cache_size, args.beam_size, transformer)
    encoder.eval()

    # begin to export encoder
    chunk_xs = torch.randn(batch_size, audio_len, feature_size, dtype=torch.float32)
    chunk_lens = torch.ones(batch_size, dtype=torch.int32) * audio_len

    offset = torch.arange(0, batch_size).unsqueeze(1)
    #  (elayers, b, head, cache_t1, d_k * 2)
    head = configs["encoder_conf"]["attention_heads"]
    d_k = configs["encoder_conf"]["output_size"] // head
    att_cache = torch.randn(batch_size, num_layers, head,
                            required_cache_size, d_k * 2,
                            dtype=torch.float32)
    cnn_cache = torch.randn(batch_size, num_layers, output_size,
                            cnn_module_kernel, dtype=torch.float32)

    cache_mask = torch.ones(batch_size, 1, required_cache_size, dtype=torch.float32)
    input_names = ['chunk_xs', 'chunk_lens', 'offset',
                   'att_cache', 'cnn_cache', 'cache_mask']
    output_names = ['log_probs', 'log_probs_idx', 'chunk_out',
                    'chunk_out_lens', 'r_offset', 'r_att_cache',
                    'r_cnn_cache', 'r_cache_mask']
    input_tensors = (chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask)
    if transformer:
        output_names.pop(6)

    all_names = input_names + output_names
    dynamic_axes = {}
    for name in all_names:
        # only the first dimension is dynamic
        # all other dimension is fixed
        dynamic_axes[name] = {0: 'B'}

    torch.onnx.export(encoder,
                      input_tensors,
                      encoder_onnx_path,
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      verbose=False)

    with torch.no_grad():
        torch_outs = encoder(chunk_xs, chunk_lens, offset,
                             att_cache, cnn_cache, cache_mask)
    if transformer:
        torch_outs = list(torch_outs).pop(6)
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path,
                                               providers=["CUDAExecutionProvider"])
    ort_inputs = {}

    input_tensors = to_numpy(input_tensors)
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]
    if transformer:
        del ort_inputs['cnn_cache']
    ort_outs = ort_session.run(None, ort_inputs)
    test(to_numpy(torch_outs), ort_outs, rtol=1e-03, atol=1e-05)
    logger.info("export to onnx streaming encoder succeed!")
    onnx_config = {
        "subsampling_rate": subsampling,
        "context": context,
        "decoding_chunk_size": decoding_chunk_size,
        "num_decoding_left_chunks": num_decoding_left_chunks,
        "beam_size": args.beam_size,
        "fp16": args.fp16,
        "feat_size": feature_size,
        "decoding_window": decoding_window,
        "cnn_module_kernel_cache": cnn_module_kernel
    }
    return onnx_config

def export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path):
    bz, seq_len = 32, 100
    beam_size = args.beam_size
    decoder = Decoder(model.decoder,
                      model.ctc_weight,
                      model.reverse_weight,
                      beam_size)
    decoder.eval()

    hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(low=3, high=seq_len, size=(bz, beam_size),
                                  dtype=torch.int32)
    r_hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3, high=seq_len, size=(bz,), dtype=torch.int32)
    ctc_score = torch.randn(bz, beam_size, dtype=torch.float32)

    input_names = ['encoder_out', 'encoder_out_lens',
                   'hyps_pad_sos_eos', 'hyps_lens_sos',
                   'r_hyps_pad_sos_eos', 'ctc_score']

    torch.onnx.export(decoder,
                      (encoder_out, encoder_out_lens,
                       hyps_pad_sos_eos, hyps_lens_sos,
                       r_hyps_pad_sos_eos, ctc_score),
                      decoder_onnx_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=input_names,
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
        o0 = decoder(encoder_out,
                     encoder_out_lens,
                     hyps_pad_sos_eos,
                     hyps_lens_sos,
                     r_hyps_pad_sos_eos,
                     ctc_score)
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(decoder_onnx_path,
                                               providers=providers)

    input_tensors = [encoder_out, encoder_out_lens, hyps_pad_sos_eos,
                     hyps_lens_sos, r_hyps_pad_sos_eos, ctc_score]
    ort_inputs = {}
    input_tensors = to_numpy(input_tensors)
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]

    # if model.reverse weight == 0,
    # the r_hyps_pad will be removed
    # from the onnx decoder since it doen't play any role
    if model.reverse_weight == 0:
        del ort_inputs['r_hyps_pad_sos_eos']
    ort_outs = ort_session.run(None, ort_inputs)

    # check decoder output
    test(to_numpy([o0]), ort_outs, rtol=1e-03, atol=1e-05)
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
    # arguments for streaming encoder
    parser.add_argument('--streaming',
                        action='store_true',
                        help="whether to export streaming encoder, default false")
    parser.add_argument('--decoding_chunk_size',
                        default=16,
                        type=int,
                        required=False,
                        help='the decoding chunk size, <=0 is not supported')
    parser.add_argument('--num_decoding_left_chunks',
                        default=5,
                        type=int,
                        required=False,
                        help="number of left chunks, <= 0 is not supported")
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
    export_enc_func = None
    if args.streaming:
        assert args.decoding_chunk_size > 0
        assert args.num_decoding_left_chunks > 0
        export_enc_func = export_online_encoder
    else:
        export_enc_func = export_offline_encoder

    onnx_config = export_enc_func(model, configs, args, logger, encoder_onnx_path)

    decoder_onnx_path = os.path.join(args.output_onnx_dir, 'decoder.onnx')
    export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path)

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

    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
