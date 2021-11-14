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

from tritonclient.utils import np_to_triton_dtype
import numpy as np
import soundfile as sf

class SpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        sequence_id = 10086 + idx
        result = ''
        inputs = [
            self.protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            self.protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype))
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        response = self.triton_client.infer(self.model_name,
                                            inputs,
                                            request_id=str(sequence_id),
                                            outputs=outputs)
        result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
        return [result]
