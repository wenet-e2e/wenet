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

import numpy as np
import json
import torch
import yaml

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from decoder import RivaWFSTOnlineDecoder


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
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

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # get device
        if args["model_instance_kind"] == "GPU":
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"])

        self.init_decoder(self.model_config["parameters"])

        print("Finish Init")

    def init_decoder(self, parameters):
        blank_id = 0
        vocab_path = None
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "blank_id":
                blank_id = int(value)
            elif key == "vocabulary":
                vocab_path = value
            elif key == "tlg_decoding_config":
                with open(str(value), "rb") as f:
                    self.tlg_decoding_config = yaml.load(f, Loader=yaml.Loader)
            elif key == "tlg_dir":
                self.tlg_dir = str(value)
            elif key == "decoding_method":
                self.decoding_method = str(value)

        self.blank_id = blank_id
        _, vocab = self.load_vocab(vocab_path)

        self.vocabulary = vocab
        self.sos = self.eos = len(vocab) - 1

        if "tlg" in self.decoding_method:
            self.decoder = RivaWFSTOnlineDecoder(len(self.vocabulary),
                                                 self.tlg_dir,
                                                 self.tlg_decoding_config)

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

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        batch_log_probs, batch_len = [], []
        batch_corr_ids, batch_start, batch_end = [], [], []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for batch_idx, request in enumerate(requests):
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "ctc_log_probs")
            ctc_log_probs = from_dlpack(in_0.to_dlpack())
            assert ctc_log_probs.shape[0] == 1, "Only support batch size 1"
            batch_log_probs.append(ctc_log_probs[0])

            in_3 = pb_utils.get_input_tensor_by_name(request, "chunk_out_lens")
            batch_len.append(from_dlpack(in_3.to_dlpack()))

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            batch_start.append(True if start else False)

            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            assert ready

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            batch_corr_ids.append(corrid)

            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]
            batch_end.append(True if end else False)

            if start:
                # intialize states
                if self.decoding_method == "tlg":
                    success = self.decoder.initialize_sequence(corrid)
                    assert success

        ctc_log_probs = torch.stack(batch_log_probs, dim=0).to(self.device)
        encoder_out_lens = torch.cat(batch_len, dim=0).squeeze(-1)

        if self.decoding_method == "tlg":
            total_hyps = self.decoder.decode_batch(
                ctc_log_probs,
                encoder_out_lens,
                batch_corr_ids,
                batch_start,
                batch_end,
            )
        else:
            raise NotImplementedError

        responses = []
        for sentence in total_hyps:
            sent = np.array(sentence)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           sent.astype(self.output0_dtype))
            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(response)
        assert len(requests) == len(responses)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
        del self.model
