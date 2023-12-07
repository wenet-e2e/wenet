# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import triton_python_backend_utils as pb_utils
import numpy as np

import torch
from torch.utils.dlpack import from_dlpack
import json
import os
import yaml
from decoder import RivaWFSTDecoder, ctc_greedy_search


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.feature_size = encoder_config['dims'][-1]

        self.lm = None
        self.init_decoder(self.model_config['parameters'])
        print('Initialized Scoring Module!')

    def init_decoder(self, parameters):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        cutoff_prob = 0.9999
        blank_id = 0
        alpha = 2.0
        beta = 1.0
        ignore_id = -1
        bidecoder = 0
        lm_path, vocab_path = None, None
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "num_processes":
                num_processes = int(value)
            elif key == "blank_id":
                blank_id = int(value)
            elif key == "cutoff_prob":
                cutoff_prob = float(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "alpha":
                alpha = float(value)
            elif key == "beta":
                beta = float(value)
            elif key == "vocabulary":
                vocab_path = value
            elif key == 'ignore_id':
                ignore_id = int(value)
            elif key == "bidecoder":
                bidecoder = int(value)
            elif key == "tlg_decoding_config":
                with open(str(value), 'rb') as f:
                    self.tlg_decoding_config = yaml.load(f, Loader=yaml.Loader)
            elif key == "tlg_dir":
                self.tlg_dir = str(value)
            elif key == "decoding_method":
                self.decoding_method = str(value)
            elif key == "attention_rescoring":
                self.rescore = int(value)
            elif key == "beam_size":
                self.beam_size = int(value)

        self.num_processes = num_processes
        self.cutoff_prob = cutoff_prob
        self.blank_id = blank_id
        _, vocab = self.load_vocab(vocab_path)
        if lm_path and os.path.exists(lm_path):
            self.lm = Scorer(alpha, beta, lm_path, vocab)
            print("Successfully load language model!")
        self.vocabulary = vocab
        self.bidecoder = bidecoder
        sos = eos = len(vocab) - 1
        self.sos = sos
        self.eos = eos
        self.ignore_id = ignore_id

        if "tlg" in self.decoding_method:
            self.decoder = RivaWFSTDecoder(len(self.vocabulary), self.tlg_dir,
                                           self.tlg_decoding_config,
                                           self.beam_size)

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def collect_inputs(self, requests):
        encoder_out_list, encoder_out_lens_list, ctc_log_probs_list, batch_count_list = [], [], [], [] # noqa
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request,
                                                     "encoder_out_lens")
            in_2 = pb_utils.get_input_tensor_by_name(request, "ctc_log_probs")

            in_0_tensor = from_dlpack(in_0.to_dlpack())
            in_1_tensor = from_dlpack(in_1.to_dlpack())
            in_2_tensor = from_dlpack(in_2.to_dlpack())

            encoder_out_list.append(in_0_tensor)
            encoder_out_lens_list.append(in_1_tensor)
            ctc_log_probs_list.append(in_2_tensor)

            batch_count_list.append(in_0_tensor.shape[0])

        encoder_tensors, logits_tensors = [], []
        for encoder_tensor, logits_tensor in zip(encoder_out_list,
                                                 ctc_log_probs_list):
            encoder_tensors += [
                item.squeeze(0) for item in encoder_tensor.split(1)
            ]
            logits_tensors += [
                item.squeeze(0) for item in logits_tensor.split(1)
            ]
        encoder_out = torch.nn.utils.rnn.pad_sequence(encoder_tensors,
                                                      batch_first=True,
                                                      padding_value=0.0)
        logits = torch.nn.utils.rnn.pad_sequence(logits_tensors,
                                                 batch_first=True,
                                                 padding_value=0.0)
        encoder_out_len = torch.cat(encoder_out_lens_list, dim=0)
        return encoder_out, encoder_out_len, logits, batch_count_list

    def rescore_hyps(self, total_tokens, nbest_scores, max_hyp_len,
                     encoder_out, encoder_out_len):
        """
        Rescore the hypotheses with attention rescoring
        """
        # TODO: need separated am_score, lm_score
        # https://github.com/k2-fsa/icefall/blob/master/icefall/decode.py#L1072-L1075
        raise NotImplementedError

    def prepare_response(self, hyps, batch_count_list):
        """
        Prepare the response
        """
        responses = []
        st = 0
        for b in batch_count_list:
            sents = np.array(hyps[st:st + b])
            out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
            st += b
        return responses

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        encoder_out, encoder_out_len, ctc_log_probs, batch_count = self.collect_inputs(
            requests)  # noqa
        ctc_log_probs = ctc_log_probs.cuda()
        if self.decoding_method == "tlg_mbr":
            total_hyps = self.decoder.decode_mbr(ctc_log_probs,
                                                 encoder_out_len)
        elif self.decoding_method == "ctc_greedy_search":
            total_hyps = ctc_greedy_search(ctc_log_probs, encoder_out_len,
                                           self.vocabulary, self.blank_id,
                                           self.eos)
        elif self.decoding_method == "tlg":
            nbest_hyps, nbest_ids, nbest_scores, max_hyp_len = self.decoder.decode_nbest( \
                ctc_log_probs, encoder_out_len)  # noqa
            total_hyps = [nbest[0] for nbest in nbest_hyps]

        if self.decoding_method == "tlg" and self.rescore:
            assert self.beam_size > 1, "Beam size must be greater than 1 for rescoring"
            selected_ids = self.rescore_hyps(nbest_ids, nbest_scores,
                                             max_hyp_len, encoder_out,
                                             encoder_out_len)
            total_hyps = [
                nbest[i] for nbest, i in zip(nbest_hyps, selected_ids)
            ]

        responses = self.prepare_response(total_hyps, batch_count)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
