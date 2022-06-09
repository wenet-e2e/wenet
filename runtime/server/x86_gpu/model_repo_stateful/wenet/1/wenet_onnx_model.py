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


import multiprocessing
import numpy as np
import os
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
from swig_decoders import ctc_beam_search_decoder_batch, Scorer, map_batch

class WenetModel(object):
    def __init__(self, model_config, device):
        params = self.parse_model_parameters(model_config)

        self.device = device
        print("Using device", device)
        print("Successfully load model !")

        # load vocabulary
        ret = self.load_vocab(params["vocab_path"])
        self.id2vocab, self.vocab, space_id, blank_id, sos_eos = ret
        self.space_id = space_id if space_id else -1
        self.blank_id = blank_id if blank_id else 0
        self.eos = self.sos = sos_eos if sos_eos else len(self.vocab) - 1
        print("Successfully load vocabulary !")
        self.params = params

        # beam search setting
        self.beam_size = params.get("beam_size")
        self.cutoff_prob = params.get("cutoff_prob")

        # language model
        lm_path = params.get("lm_path", None)
        alpha, beta = params.get('alpha'), params.get('beta')
        self.scorer = None
        if os.path.exists(lm_path):
            self.scorer = Scorer(alpha, beta, lm_path, self.vocab)

        self.bidecoder = params.get('bidecoder')
        # rescore setting
        self.rescoring = params.get("rescoring", 0)
        print("Using rescoring:", bool(self.rescoring))
        print("Successfully load all parameters!")

        self.dtype = torch.float16

    def generate_init_cache(self):
        encoder_out = None
        return encoder_out

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        space_id, blank_id, sos_eos = None, None, None
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
                if char == " ":
                    space_id = int(id)
                elif char == "<blank>":
                    blank_id = int(id)
                elif char == "<sos/eos>":
                    sos_eos = int(id)
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return (id2vocab, vocab, space_id, blank_id, sos_eos)

    def parse_model_parameters(self, model_parameters):
        model_p = {"beam_size": 10,
                   "cutoff_prob": 0.999,
                   "vocab_path": None,
                   "lm_path": None,
                   "alpha": 2.0,
                   "beta": 1.0,
                   "rescoring": 0,
                   "bidecoder": 1}
        # get parameter configurations
        for li in model_parameters.items():
            key, value = li
            true_value = value["string_value"]
            if key not in model_p:
                continue
            key_type = type(model_p[key])
            if key_type == type(None):
                model_p[key] = true_value
            else:
                model_p[key] = key_type(true_value)
        assert model_p["vocab_path"] is not None
        return model_p

    def infer(self, batch_log_probs, batch_log_probs_idx,
              seq_lens, rescore_index, batch_states):
        """
        batch_states = [trieVector, batch_start,
                       batch_encoder_hist, cur_encoder_out]
        """
        trie_vector, batch_start, batch_encoder_hist, cur_encoder_out = batch_states
        num_processes = min(multiprocessing.cpu_count(), len(batch_log_probs))

        score_hyps = self.batch_ctc_prefix_beam_search_cpu(batch_log_probs,
                                                           batch_log_probs_idx,
                                                           seq_lens,
                                                           trie_vector,
                                                           batch_start,
                                                           self.beam_size,
                                                           self.blank_id,
                                                           self.space_id,
                                                           self.cutoff_prob,
                                                           num_processes,
                                                           self.scorer)

        if self.rescoring and len(rescore_index) != 0:
            # find the end of sequence
            rescore_encoder_hist = []
            rescore_encoder_lens = []
            rescore_hyps = []
            res_idx = list(rescore_index.keys())
            max_length = -1
            for idx in res_idx:
                hist_enc = batch_encoder_hist[idx]
                if hist_enc is None:
                    cur_enc = cur_encoder_out[idx]
                else:
                    cur_enc = torch.cat([hist_enc, cur_encoder_out[idx]], axis=0)
                rescore_encoder_hist.append(cur_enc)
                cur_mask_len = int(len(hist_enc) + seq_lens[idx])
                rescore_encoder_lens.append(cur_mask_len)
                rescore_hyps.append(score_hyps[idx])
                if cur_enc.shape[0] > max_length:
                    max_length = cur_enc.shape[0]
            best_index = self.batch_rescoring(rescore_hyps, rescore_encoder_hist,
                                              rescore_encoder_lens, max_length)

        best_sent = []
        j = 0
        for idx, li in enumerate(score_hyps):
            if idx in rescore_index and self.rescoring:
                best_sent.append(li[best_index[j]][1])
                j += 1
            else:
                best_sent.append(li[0][1])

        final_result = map_batch(best_sent, self.vocab, num_processes)

        return final_result, cur_encoder_out

    def batch_ctc_prefix_beam_search_cpu(self, batch_log_probs_seq,
                                         batch_log_probs_idx,
                                         batch_len, batch_root,
                                         batch_start, beam_size,
                                         blank_id, space_id,
                                         cutoff_prob, num_processes,
                                         scorer):
        """
        Return: Batch x Beam_size elements, each element is a tuple
                (score, list of ids),
        """

        batch_len_list = batch_len
        batch_log_probs_seq_list = []
        batch_log_probs_idx_list = []
        for i in range(len(batch_len_list)):
            cur_len = int(batch_len_list[i])
            batch_log_probs_seq_list.append(batch_log_probs_seq[i][0:cur_len].tolist())
            batch_log_probs_idx_list.append(batch_log_probs_idx[i][0:cur_len].tolist())
        score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq_list,
                                                   batch_log_probs_idx_list,
                                                   batch_root,
                                                   batch_start,
                                                   beam_size,
                                                   num_processes,
                                                   blank_id,
                                                   space_id,
                                                   cutoff_prob,
                                                   scorer)
        return score_hyps

    def batch_rescoring(self, score_hyps, hist_enc, hist_mask_len, max_len):
        """
        score_hyps: [((ctc_score, (id1, id2, id3, ....)), (), ...), ....]
        hist_enc: [len1xF, len2xF, .....]
        hist_mask: [1x1xlen1, 1x1xlen2]
        return bzx1  best_index
        """
        bz = len(hist_enc)
        f = hist_enc[0].shape[-1]
        beam_size = self.beam_size
        encoder_lens = np.zeros((bz, 1), dtype=np.int32)
        encoder_out = torch.zeros((bz, max_len, f), dtype=self.dtype)
        hyps = []
        ctc_score = torch.zeros((bz, beam_size), dtype=self.dtype)
        max_seq_len = 0
        for i in range(bz):
            cur_len = hist_enc[i].shape[0]
            encoder_out[i, 0:cur_len] = hist_enc[i]
            encoder_lens[i, 0] = hist_mask_len[i]

            # process candidate
            if len(score_hyps[i]) < beam_size:
                to_append = (beam_size - len(score_hyps[i])) * [(-10000, ())]
                score_hyps[i] = list(score_hyps[i]) + to_append
            for idx, c in enumerate(score_hyps[i]):
                score, idlist = c
                if score < -10000:
                    score = -10000
                ctc_score[i][idx] = score
                hyps.append(list(idlist))
                if len(hyps[-1]) > max_seq_len:
                    max_seq_len = len(hyps[-1])

        max_seq_len += 2
        hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len), dtype=np.int64)
        hyps_pad_sos_eos = hyps_pad_sos_eos * self.eos  # fill eos
        if self.bidecoder:
            r_hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len), dtype=np.int64)
            r_hyps_pad_sos_eos = r_hyps_pad_sos_eos * self.eos

        hyps_lens_sos = np.ones((bz, beam_size), dtype=np.int32)
        bz_id = 0
        for idx, cand in enumerate(hyps):
            bz_id = idx // beam_size
            length = len(cand) + 2
            bz_offset = idx % beam_size
            pad_cand = [self.sos] + cand + [self.eos]
            hyps_pad_sos_eos[bz_id][bz_offset][0 : length] = pad_cand
            if self.bidecoder:
                r_pad_cand = [self.sos] + cand[::-1] + [self.eos]
                r_hyps_pad_sos_eos[bz_id][bz_offset][0:length] = r_pad_cand
            hyps_lens_sos[bz_id][idx % beam_size] = len(cand) + 1
        in0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(encoder_out))
        in1 = pb_utils.Tensor("encoder_out_lens", encoder_lens)
        in2 = pb_utils.Tensor("hyps_pad_sos_eos", hyps_pad_sos_eos)
        in3 = pb_utils.Tensor("hyps_lens_sos", hyps_lens_sos)
        input_tensors = [in0, in1, in2, in3]
        if self.bidecoder:
            in4 = pb_utils.Tensor("r_hyps_pad_sos_eos", r_hyps_pad_sos_eos)
            input_tensors.append(in4)
        in5 = pb_utils.Tensor.from_dlpack("ctc_score", to_dlpack(ctc_score))
        input_tensors.append(in5)
        request = pb_utils.InferenceRequest(model_name='decoder',
                                            requested_output_names=['best_index'],
                                            inputs=input_tensors)
        response = request.exec()
        best_index = pb_utils.get_output_tensor_by_name(response, 'best_index')
        best_index = from_dlpack(best_index.to_dlpack()).clone()
        best_index = best_index.numpy()[:, 0]
        return best_index

    def __del__(self):
        print("remove wenet model")
