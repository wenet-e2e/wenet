# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import numpy as np

import torch
import torchaudio.functional as F


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def replace_duplicates_with_blank(hyp: List[int],
                                  blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(
                hyp) and hyp[cur] == hyp[prev] and hyp[cur] != blank_id:
            new_hyp.append(blank_id)
            cur += 1
    return new_hyp


def gen_ctc_peak_time(hyp: List[int], blank_id: int = 0) -> List[int]:
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            times.append(cur)
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return times


def gen_timestamps_from_peak(
    peaks: List[int],
    max_duration: float,
    frame_rate: float = 0.04,
    max_token_duration: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Args:
        peaks: ctc peaks time stamp
        max_duration: max_duration of the sentence
        frame_rate: frame rate of every time stamp, in seconds
        max_token_duration: max duration of the token, in seconds
    Returns:
        list(start, end) of each token
    """
    times = []
    half_max = max_token_duration / 2
    for i in range(len(peaks)):
        if i == 0:
            start = max(0, peaks[0] * frame_rate - half_max)
        else:
            start = max((peaks[i - 1] + peaks[i]) / 2 * frame_rate,
                        peaks[i] * frame_rate - half_max)

        if i == len(peaks) - 1:
            end = min(max_duration, peaks[-1] * frame_rate + half_max)
        else:
            end = min((peaks[i] + peaks[i + 1]) / 2 * frame_rate,
                      peaks[i] * frame_rate + half_max)
        times.append((start, end))
    return times


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


def force_align(ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
    """ctc forced alignment.

    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    ctc_probs = ctc_probs[None].cpu()
    y = y[None].cpu()
    alignments, _ = F.forced_align(ctc_probs, y, blank=blank_id)
    return alignments[0]


def get_blank_id(configs, symbol_table):
    if 'ctc_conf' not in configs:
        configs['ctc_conf'] = {}

    if '<blank>' in symbol_table:
        if 'ctc_blank_id' in configs['ctc_conf']:
            assert configs['ctc_conf']['ctc_blank_id'] == symbol_table[
                '<blank>']
        else:
            configs['ctc_conf']['ctc_blank_id'] = symbol_table['<blank>']
    else:
        assert 'ctc_blank_id' in configs[
            'ctc_conf'], "PLZ set ctc_blank_id in yaml"

    return configs, configs['ctc_conf']['ctc_blank_id']
