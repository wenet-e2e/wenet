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

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import torch
import kaldifeat
import _kaldifeat
from typing import List
import json
import numpy as np

class Fbank(torch.nn.Module):
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)

class Feat(object):
    def __init__(self, seqid, offset_ms, sample_rate,
                 first_chunk_sz, frame_stride, device='cpu'):
        self.seqid = seqid
        self.sample_rate = sample_rate
        self.wav = torch.tensor([], device=device)
        self.offset = int(offset_ms / 1000 * sample_rate)
        self.frames = None
        self.frame_stride = int(frame_stride)
        self.first_chunk_sz = first_chunk_sz
        self.device = device

    def add_wavs(self, wav: torch.tensor):
        if len(self.wav) == 0 and len(wav) < self.first_chunk_sz:
            raise Exception("Invalid first chunk size", len(wav))
        wav = wav.to(self.device)
        self.wav = torch.cat([self.wav, wav], axis=0)

    def get_seg_wav(self):
        seg = self.wav[:]
        self.wav = self.wav[-self.offset:]
        return seg

    def add_frames(self, frames: torch.tensor):
        """
        frames: seq_len x feat_sz
        """
        if self.frames is None:
            self.frames = frames
        else:
            self.frames = torch.cat([self.frames, frames], axis=0)

    def get_frames(self, num_frames: int):
        seg = self.frames[0: num_frames]
        self.frames = self.frames[self.frame_stride:]
        return seg

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

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "speech")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        if self.output0_dtype == np.float32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self.feature_size = output0_config['dims'][-1]
        self.decoding_window = output0_config['dims'][-2]
        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "speech_lengths")
        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        feat_opt = self.parse_model_params(model_config["parameters"])

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0
        opts.mel_opts.num_bins = self.feature_size
        frame_length_ms = feat_opt["frame_length_ms"]
        frame_shift_ms = feat_opt["frame_shift_ms"]
        opts.frame_opts.frame_length_ms = frame_length_ms
        opts.frame_opts.frame_shift_ms = frame_shift_ms
        opts.frame_opts.samp_freq = feat_opt["sample_rate"]
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)
        self.seq_feat = {}
        chunk_size_s = feat_opt["chunk_size_s"]
        sample_rate = feat_opt["sample_rate"]
        self.chunk_size = int(chunk_size_s * sample_rate)
        self.frame_stride = (chunk_size_s * 1000) // frame_shift_ms

        first_chunk_size = int(self.chunk_size)
        cur_frames = _kaldifeat.num_frames(first_chunk_size, opts.frame_opts)
        while cur_frames < self.decoding_window:
            first_chunk_size += frame_shift_ms * sample_rate // 1000
            cur_frames = _kaldifeat.num_frames(first_chunk_size, opts.frame_opts)
        #  self.pad_silence = first_chunk_size - self.chunk_size
        self.first_chunk_size = first_chunk_size
        self.offset_ms = self.get_offset(frame_length_ms, frame_shift_ms)
        self.sample_rate = sample_rate
        self.min_seg = frame_length_ms * sample_rate // 1000
        print("MIN SEG IS", self.min_seg)

    def get_offset(self, frame_length_ms, frame_shift_ms):
        offset_ms = 0
        while offset_ms + frame_shift_ms < frame_length_ms:
            offset_ms += frame_shift_ms
        return offset_ms

    def parse_model_params(self, model_params):
        model_p = {
            "frame_length_ms": 25,
            "frame_shift_ms": 10,
            "sample_rate": 16000,
            "chunk_size_s": 0.64}
        # get parameter configurations
        for li in model_params.items():
            key, value = li
            true_value = value["string_value"]
            if key not in model_p:
                continue
            key_type = type(model_p[key])
            if key_type == type(None):
                model_p[key] = true_value
            else:
                model_p[key] = key_type(true_value)
        return model_p

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
        total_waves = []
        responses = []
        batch_seqid = []
        end_seqid = {}
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            #  wavs = input0.as_numpy()[0]
            wavs = from_dlpack(input0.to_dlpack())[0]

            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")
            #  wav_lens = input1.as_numpy()[0][0]
            wav_lens = from_dlpack(input1.to_dlpack())[0]

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            if start:
                self.seq_feat[corrid] = Feat(corrid, self.offset_ms,
                                             self.sample_rate,
                                             self.first_chunk_size,
                                             self.frame_stride,
                                             self.device)
            if ready:
                self.seq_feat[corrid].add_wavs(wavs[0:wav_lens])

            batch_seqid.append(corrid)
            if end:
                end_seqid[corrid] = 1

            # if not start
            # check chunk ms size

            wav = self.seq_feat[corrid].get_seg_wav() * 32768
            if len(wav) < self.min_seg:
                temp = torch.zeros(self.min_seg, dtype=torch.float32,
                                   device=self.device)
                temp[0:len(wav)] = wav[:]
                wav = temp
            total_waves.append(wav)

        features = self.feature_extractor(total_waves)

        batch_size = len(batch_seqid)
        batch_speech = torch.zeros((batch_size, self.decoding_window,
                                    self.feature_size), dtype=self.dtype)
        batch_speech_lens = torch.zeros((batch_size, 1), dtype=torch.int32)
        i = 0
        for corrid, frames in zip(batch_seqid, features):
            self.seq_feat[corrid].add_frames(frames)
            r_frames = self.seq_feat[corrid].get_frames(self.decoding_window)
            speech = batch_speech[i: i + 1]
            speech_lengths = batch_speech_lens[i: i + 1]
            i += 1
            speech_lengths[0] = r_frames.size(0)
            speech[0][0:r_frames.size(0)] = r_frames.to(speech.device)
            # out_tensor0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            # out_tensor1 = pb_utils.Tensor.from_dlpack("speech_lengths",
            #                                            to_dlpack(speech_lengths))
            out_tensor0 = pb_utils.Tensor("speech", speech.numpy())
            out_tensor1 = pb_utils.Tensor("speech_lengths", speech_lengths.numpy())
            output_tensors = [out_tensor0, out_tensor1]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
            if corrid in end_seqid:
                del self.seq_feat[corrid]
        return responses

    def finalize(self):
        print("Remove feature extractor!")
