# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import argparse
import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from wenet.cli.hub import Hub
from wenet.utils.ctc_utils import (force_align, gen_ctc_peak_time,
                                   gen_timestamps_from_peak)
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.init_model import init_model
from wenet.transformer.search import (attention_rescoring,
                                      ctc_prefix_beam_search, DecodeResult)
from wenet.utils.context_graph import ContextGraph
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu


class Model:

    def __init__(self,
                 model_dir: str,
                 gpu: int = -1,
                 beam: int = 5,
                 context_path: str = None,
                 context_score: float = 6.0,
                 resample_rate: int = 16000):
        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        self.resample_rate = resample_rate
        self.model.eval()
        if gpu >= 0:
            device = 'cuda:{}'.format(gpu)
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.model.to(device)
        self.symbol_table = read_symbol_table(units_path)
        self.char_dict = {v: k for k, v in self.symbol_table.items()}
        self.beam = beam
        if context_path is not None:
            self.context_graph = ContextGraph(context_path,
                                              self.symbol_table,
                                              context_score=context_score)
        else:
            self.context_graph = None

    def compute_feats(self, audio_file: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        # NOTE (MengqingCao): complex dtype not supported in torch_npu.abs() now,
        # thus, delay placing data on NPU after the calculation of fbank.
        # revert me after complex dtype is supported.
        if "npu" not in self.device.__str__():
            waveform = waveform.to(self.device)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        if "npu" in self.device.__str__():
            feats = feats.to(self.device)
        feats = feats.unsqueeze(0)
        return feats

    @torch.no_grad()
    def _decode(self,
                audio_file: str,
                tokens_info: bool = False,
                label: str = None) -> dict:
        feats = self.compute_feats(audio_file)
        encoder_out, _, _ = self.model.forward_encoder_chunk(feats, 0, -1)
        encoder_lens = torch.tensor([encoder_out.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)
        ctc_probs = self.model.ctc_activation(encoder_out)
        if label is None:
            ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph)
        else:  # force align mode, construct ctc prefix result from alignment
            label_t = self.tokenize(label)
            alignment = force_align(ctc_probs.squeeze(0),
                                    torch.tensor(label_t, dtype=torch.long))
            peaks = gen_ctc_peak_time(alignment)
            ctc_prefix_results = [
                DecodeResult(tokens=label_t,
                             score=0.0,
                             times=peaks,
                             nbest=[label_t],
                             nbest_scores=[0.0],
                             nbest_times=[peaks])
            ]
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence

        if tokens_info:
            frame_rate = self.model.subsampling_rate(
            ) * 0.01  # 0.01 seconds per frame
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(res.times, max_duration,
                                             frame_rate, 1.0)
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.char_dict[x],
                    'start': round(times[i][0], 3),
                    'end': round(times[i][1], 3),
                    'confidence': round(res.tokens_confidence[i], 2)
                })
            result['tokens'] = tokens_info
        return result

    def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
        return self._decode(audio_file, tokens_info)

    def tokenize(self, label: str):
        # TODO(Binbin Zhang): Support BPE
        tokens = []
        for c in label:
            if c == ' ':
                c = "▁"
            tokens.append(c)
        token_list = []
        for c in tokens:
            if c in self.symbol_table:
                token_list.append(self.symbol_table[c])
            elif '<unk>' in self.symbol_table:
                token_list.append(self.symbol_table['<unk>'])
        return token_list

    def align(self, audio_file: str, label: str) -> dict:
        return self._decode(audio_file, True, label)


def load_model(language: str = None,
               model_dir: str = None,
               gpu: int = -1,
               beam: int = 5,
               context_path: str = None,
               context_score: float = 6.0,
               device: str = "cpu") -> Model:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang(language)

    if gpu != -1:
        # remain the original usage of gpu
        device = "cuda"
    model = Model(model_dir, gpu, beam, context_path, context_score)
    model.device = torch.device(device)
    model.model.to(device)
    return model

# Load the pytorch pt model which contains all the details compared with jit.
# And we can use the pt model as a third party pytorch nn.Module for training
def load_model_pt(model_dir):
    """ There are the followi files in in `model_dir`
        * final.pt, required
        * train.yaml, required
        * units.txt, required
        * global_cmvn, optional
    """
    # Check required files
    required_files = ['train.yaml', 'final.pt', 'units.txt']
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file {file} not found in {model_dir}")
    # Read config and override some config
    config_file = os.path.join(model_dir, 'train.yaml')
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    token_file = os.path.join(model_dir, 'units.txt')
    configs['tokenizer_conf']['symbol_table_path'] = token_file
    cmvn_file = os.path.join(model_dir, 'global_cmvn')
    if os.path.exists(cmvn_file):
        configs['cmvn_conf']['cmvn_file'] = cmvn_file
    # Read model
    pt_file = os.path.join(model_dir, 'final.pt')
    args = argparse.Namespace()
    args.checkpoint = pt_file
    model, configs = init_model(args, configs)
    return model
