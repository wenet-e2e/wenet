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
from swig_decoders import PathTrie, TrieVector

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from wenet_onnx_model import WenetModel

from torch.utils.dlpack import from_dlpack

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
        self.model_config = model_config = json.loads(args['model_config'])

        # get device
        if args["model_instance_kind"] == "GPU":
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # get parameter configurations
        self.model = WenetModel(model_config["parameters"], self.device)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # use to record every sequence state
        self.seq_states = {}
        print("Finish Init")

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
        responses = []
        batch_log_probs, batch_log_probs_idx, batch_len, batch_states = [], [], [], []
        cur_encoder_out = []

        batch_encoder_hist = []
        batch_start = []

        trieVector = TrieVector()

        rescore_index = {}
        batch_idx2_corrid = {}

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        batch_idx = 0
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "log_probs")
            batch_log_probs.append(in_0.as_numpy()[0])
            in_1 = pb_utils.get_input_tensor_by_name(request, "log_probs_idx")
            batch_log_probs_idx.append(in_1.as_numpy()[0])
            if self.model.rescoring:
                in_2 = pb_utils.get_input_tensor_by_name(request, "chunk_out")
                # important to use clone or this tensor
                # the tensor will be released after one inference
                in_2 = from_dlpack(in_2.to_dlpack()).clone()
                cur_encoder_out.append(in_2[0])
            in_3 = pb_utils.get_input_tensor_by_name(request, "chunk_out_lens")
            batch_len.append(in_3.as_numpy())

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]

            if start:
                batch_start.append(True)
            else:
                batch_start.append(False)

            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]

            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            if start and ready:
                # intialize states
                encoder_out = self.model.generate_init_cache()
                root = PathTrie()
                # register this sequence
                self.seq_states[corrid] = [root, encoder_out]

            if end and ready:
                rescore_index[batch_idx] = 1

            if ready:
                root, encoder_out = self.seq_states[corrid]
                trieVector.append(root)
                batch_idx2_corrid[batch_idx] = corrid
                batch_encoder_hist.append(encoder_out)

            batch_idx += 1

        batch_states = [trieVector, batch_start, batch_encoder_hist, cur_encoder_out]
        res_sents, new_states = self.model.infer(batch_log_probs, batch_log_probs_idx,
                                                 batch_len, rescore_index, batch_states)
        cur_encoder_out = new_states
        for i in range(len(res_sents)):
            sent = np.array(res_sents[i])
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", sent.astype(self.output0_dtype))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(response)
            corr = batch_idx2_corrid[i]
            if i in rescore_index:
                # this response ends, remove it
                del self.seq_states[corr]
            else:
                if self.model.rescoring:
                    if self.seq_states[corr][1] is None:
                        self.seq_states[corr][1] = cur_encoder_out[i]
                    else:
                        new_hist = torch.cat([self.seq_states[corr][1],
                                              cur_encoder_out[i]], axis=0)
                        self.seq_states[corr][1] = new_hist

        assert len(requests) == len(responses)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        del self.model
