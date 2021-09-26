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

from __future__ import print_function

import argparse
import copy
import logging
import multiprocessing
import os
import yaml
import numpy as np
import onnxruntime

import torch
from torch.utils.data import DataLoader

from wenet.utils.common import IGNORE_ID
from wenet.dataset.dataset import AudioDataset, CollateFunc
from swig_decoders import map_batch, ctc_beam_search_decoder_batch, TrieVector, PathTrie

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument(
        '--encoder_onnx',
        required=True,
        help='encoder onnx file')
    parser.add_argument(
        '--decoder_onnx',
        required=True,
        help='decoder onnx file')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    if raw_wav:
        test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    test_collate_func = CollateFunc(**test_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data,
                                **dataset_conf,
                                raw_wav=raw_wav)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    # Init asr model from configs
    encoder_ort_session = onnxruntime.InferenceSession(args.encoder_onnx)
    decoder_ort_session = None
    if args.mode == "attention_rescoring":
        decoder_ort_session = onnxruntime.InferenceSession(args.decoder_onnx)

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
    batch_size = args.batch_size
    num_processes = min(multiprocessing.cpu_count(), batch_size)
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats
            feats_lengths = feats_lengths
            ort_inputs = {
                encoder_ort_session.get_inputs()[0].name: feats.numpy(),
                encoder_ort_session.get_inputs()[1].name: feats_lengths.numpy()}
            ort_outs = encoder_ort_session.run(None, ort_inputs)
            encoder_out, encoder_out_lens, batch_log_probs, \
                batch_log_probs_idx = ort_outs
            beam_size = batch_log_probs.shape[-1]
            hyps, score_hyps = [], []
            if args.mode == 'ctc_greedy_search':
                assert batch_log_probs.shape[-1] == 1
                batch_sents = []
                for idx, seq in enumerate(batch_log_probs_idx):
                    batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                hyps = map_batch(
                    batch_sents,
                    vocabulary,
                    num_processes,
                    greedy=True,
                    blank_id=0)
            elif args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
                batch_log_probs_seq_list = batch_log_probs.tolist()
                batch_log_probs_idx_list = batch_log_probs_idx.tolist()
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
                    for cand_hyps in score_hyps:
                        hyps.append(cand_hyps[0][1])
                    hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
            if args.mode == 'attention_rescoring':
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) != beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    for hyp in hyps:
                        ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) + 1 > max_len:
                            max_len = len(hyp[1]) + 1
                assert len(ctc_score) == beam_size * batch_size
                ctc_score = np.array(ctc_score, dtype=np.float32)
                ctc_score = ctc_score.reshape(batch_size, beam_size)
                hyps_pad_sos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                hyps_pad_eos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                r_hyps_pad_sos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                r_hyps_pad_eos = np.ones(
                    (batch_size, beam_size, max_len), dtype=np.int64) * IGNORE_ID
                hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                for i in range(batch_size):
                    for j in range(beam_size):
                        cand = all_hyps.pop(0)
                        hyps_pad_sos[i][j][0:len(cand) + 1] = [sos] + cand
                        hyps_pad_eos[i][j][0:len(cand) + 1] = cand + [eos]
                        r_hyps_pad_sos[i][j][0:len(cand) + 1] = [sos] + cand[::-1]
                        r_hyps_pad_eos[i][j][0:len(cand) + 1] = cand[::-1] + [eos]
                        hyps_lens_sos[i][j] = len(cand) + 1

                decoder_ort_inputs = {
                    decoder_ort_session.get_inputs()[0].name: encoder_out,
                    decoder_ort_session.get_inputs()[1].name: encoder_out_lens,
                    decoder_ort_session.get_inputs()[2].name: hyps_pad_sos,
                    decoder_ort_session.get_inputs()[3].name: hyps_pad_eos,
                    decoder_ort_session.get_inputs()[4].name: hyps_lens_sos,
                    decoder_ort_session.get_inputs()[5].name: r_hyps_pad_sos,
                    decoder_ort_session.get_inputs()[6].name: r_hyps_pad_eos,
                    decoder_ort_session.get_inputs()[7].name: ctc_score}
                hyps = []
                best_hyps, best_lens = decoder_ort_session.run(
                    None, decoder_ort_inputs)
                for hyp, blen in zip(best_hyps, best_lens):
                    hyps.append(hyp.tolist()[0:blen])
                hyps = map_batch(hyps, vocabulary, num_processes)

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
