# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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

"""
This script is for testing exported onnx encoder and decoder.
The exported onnx models only support batch offline ASR inference.
It requires a python wrapped c++ ctc decoder.
Please install it by following:
https://github.com/Slyne/ctc_decoder.git
"""
from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.common import IGNORE_ID
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config

import onnxruntime as rt
import multiprocessing
import numpy as np

try:
    from swig_decoders import map_batch, \
        ctc_beam_search_decoder_batch, \
        TrieVector, PathTrie
except ImportError:
    print('Please install ctc decoders first by refering to\n' +
          'https://github.com/Slyne/ctc_decoder.git')
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--encoder_onnx', required=True, help='encoder onnx file')
    parser.add_argument('--decoder_onnx', required=True, help='decoder onnx file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 'ctc_prefix_beam_search',
                            'attention_rescoring'],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        EP_list = ['CPUExecutionProvider']

    encoder_ort_session = rt.InferenceSession(args.encoder_onnx, providers=EP_list)
    decoder_ort_session = None
    if args.mode == "attention_rescoring":
        decoder_ort_session = rt.InferenceSession(args.decoder_onnx, providers=EP_list)

    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for _, batch in enumerate(test_data_loader):
            keys, feats, _, feats_lengths, _ = batch
            ort_inputs = {
                encoder_ort_session.get_inputs()[0].name: feats.numpy(),
                encoder_ort_session.get_inputs()[1].name: feats_lengths.numpy()}
            ort_outs = encoder_ort_session.run(None, ort_inputs)
            encoder_out, encoder_out_lens, ctc_log_probs, \
                beam_log_probs, beam_log_probs_idx = ort_outs
            beam_size = beam_log_probs.shape[-1]
            batch_size = beam_log_probs.shape[0]
            num_processes = min(multiprocessing.cpu_count(), batch_size)
            if args.mode == 'ctc_greedy_search':
                if beam_size != 1:
                    log_probs_idx = beam_log_probs_idx[:, :, 0]
                batch_sents = []
                for idx, seq in enumerate(log_probs_idx):
                    batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                hyps = map_batch(batch_sents, vocabulary, num_processes,
                                 True, 0)
            elif args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
                batch_log_probs_seq_list = beam_log_probs.tolist()
                batch_log_probs_idx_list = beam_log_probs_idx.tolist()
                batch_len_list = encoder_out_lens.tolist()
                batch_log_probs_seq = []
                batch_log_probs_ids = []
                batch_start = []  # only effective in streaming deployment
                batch_root = TrieVector()
                root_dict = {}
                for i in range(len(batch_len_list)):
                    num_sent = batch_len_list[i]
                    batch_log_probs_seq.append(
                        batch_log_probs_seq_list[i][0:num_sent])
                    batch_log_probs_ids.append(
                        batch_log_probs_idx_list[i][0:num_sent])
                    root_dict[i] = PathTrie()
                    batch_root.append(root_dict[i])
                    batch_start.append(True)
                score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                           batch_log_probs_ids,
                                                           batch_root,
                                                           batch_start,
                                                           beam_size,
                                                           num_processes,
                                                           0, -2, 0.99999)
                if args.mode == 'ctc_prefix_beam_search':
                    hyps = []
                    for cand_hyps in score_hyps:
                        hyps.append(cand_hyps[0][1])
                    hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
            if args.mode == 'attention_rescoring':
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    for hyp in hyps:
                        ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) + 1 > max_len:
                            max_len = len(hyp[1]) + 1
                assert len(ctc_score) == beam_size * batch_size
                hyps_pad_sos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                r_hyps_pad_sos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                k = 0
                for i in range(batch_size):
                    for j in range(beam_size):
                        cand = all_hyps[k]
                        hyps_pad_sos[i][j][0:len(cand) + 1] = [sos] + cand
                        r_hyps_pad_sos[i][j][0:len(cand) + 1] = [sos] + cand[::-1]
                        hyps_lens_sos[i][j] = len(cand) + 1
                        k += 1
                decoder_ort_inputs = {
                    decoder_ort_session.get_inputs()[0].name: encoder_out,
                    decoder_ort_session.get_inputs()[1].name: encoder_out_lens,
                    decoder_ort_session.get_inputs()[2].name: hyps_pad_sos,
                    decoder_ort_session.get_inputs()[3].name: hyps_lens_sos,
                    decoder_ort_session.get_inputs()[4].name: r_hyps_pad_sos}
                decoder_out, r_decoder_out = decoder_ort_session.run(
                    None, decoder_ort_inputs)
                best_sents = []
                k = 0
                for d_o, r_d_o in zip(decoder_out, r_decoder_out):
                    # d_0 & r_d_o: beam x T x V
                    cur_best_sent = []
                    cur_best_score = -float("inf")
                    for sent_d_o, sent_r_d_o in zip(d_o, r_d_o):
                        cand = all_hyps[k] + [eos]
                        r_cand = all_hyps[k][::-1] + [eos]
                        score, r_score = 0, 0
                        for i in range(len(cand)):
                            index, r_index = cand[i], r_cand[i]
                            score += sent_d_o[i][index]
                            r_score += sent_r_d_o[i][r_index]
                        if args.reverse_weight > 0:
                            score = score * (1 - args.reverse_weight) + \
                                args.reverse_weight * r_score
                        score = score + args.ctc_weight * ctc_score[k]
                        if score > cur_best_score:
                            cur_best_sent = all_hyps[k]
                            cur_best_score = score
                        k += 1
                    best_sents.append(cur_best_sent)
                hyps = map_batch(best_sents, vocabulary, num_processes)

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))

if __name__ == '__main__':
    main()
