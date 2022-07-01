import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
import torch
import numpy as np
import kaldifeat
import _kaldifeat
from typing import List
import json

class Fbank(torch.nn.Module):
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)


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
        output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        if output0_dtype == np.float32:
            self.output0_dtype = torch.float32
        else:
            self.output0_dtype = torch.float16

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "speech_lengths")
        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        params = self.model_config['parameters']
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0

        for li in params.items():
            key, value = li
            value = value["string_value"]
            if key == "num_mel_bins":
                opts.mel_opts.num_bins = int(value)
            elif key == "frame_shift_in_ms":
                opts.frame_opts.frame_shift_ms = float(value)
            elif key == "frame_length_in_ms":
                opts.frame_opts.frame_length_ms = float(value)
            elif key == "sample_rate":
                opts.frame_opts.samp_freq = int(value)
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)
        self.feature_size = opts.mel_opts.num_bins

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
        batch_count = []
        total_waves = []
        batch_len = []
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")

            cur_b_wav = input0.as_numpy()
            cur_b_wav = cur_b_wav * (1 << 15)  # b x -1
            cur_b_wav_lens = input1.as_numpy()  # b x 1
            cur_batch = cur_b_wav.shape[0]
            cur_len = cur_b_wav.shape[1]
            batch_count.append(cur_batch)
            batch_len.append(cur_len)
            for wav, wav_len in zip(cur_b_wav, cur_b_wav_lens):
                wav_len = wav_len[0]
                wav = torch.tensor(wav[0:wav_len], dtype=torch.float32,
                                   device=self.device)
                total_waves.append(wav)

        features = self.feature_extractor(total_waves)
        idx = 0
        for b, l in zip(batch_count, batch_len):
            expect_feat_len = _kaldifeat.num_frames(l, self.opts.frame_opts)
            speech = torch.zeros((b, expect_feat_len, self.feature_size),
                                 dtype=self.output0_dtype, device=self.device)
            speech_lengths = torch.zeros((b, 1), dtype=torch.int32, device=self.device)
            for i in range(b):
                f = features[idx]
                f_l = f.shape[0]
                speech[i, 0: f_l, :] = f.to(self.output0_dtype)
                speech_lengths[i][0] = f_l
                idx += 1
            # put speech feature on device will cause empty output
            # we will follow this issue and now temporarily put it on cpu
            speech = speech.cpu()
            speech_lengths = speech_lengths.cpu()
            out0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            out1 = pb_utils.Tensor.from_dlpack("speech_lengths",
                                               to_dlpack(speech_lengths))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0, out1])
            responses.append(inference_response)
        return responses
