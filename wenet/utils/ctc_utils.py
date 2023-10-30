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


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def replace_duplicates_with_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(hyp) and hyp[cur] == hyp[prev] and hyp[cur] != 0:
            new_hyp.append(0)
            cur += 1
    return new_hyp


def gen_ctc_peak_time(hyp: List[int]) -> List[int]:
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
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
    ctc_probs = ctc_probs.cpu()
    y = y.cpu()
    y_insert_blank = insert_blank(y, blank_id)

    log_alpha = torch.zeros((ctc_probs.size(0), len(y_insert_blank)))
    log_alpha = log_alpha - float('inf')  # log of zero
    state_path = torch.zeros((ctc_probs.size(0), len(y_insert_blank)),
                             dtype=torch.int16) - 1  # state path

    # init start state
    log_alpha[0, 0] = ctc_probs[0][y_insert_blank[0]]
    log_alpha[0, 1] = ctc_probs[0][y_insert_blank[1]]

    for t in range(1, ctc_probs.size(0)):
        for s in range(len(y_insert_blank)):
            if y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[
                    s] == y_insert_blank[s - 2]:
                candidates = torch.tensor(
                    [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]])
                prev_state = [s, s - 1]
            else:
                candidates = torch.tensor([
                    log_alpha[t - 1, s],
                    log_alpha[t - 1, s - 1],
                    log_alpha[t - 1, s - 2],
                ])
                prev_state = [s, s - 1, s - 2]
            log_alpha[
                t, s] = torch.max(candidates) + ctc_probs[t][y_insert_blank[s]]
            state_path[t, s] = prev_state[torch.argmax(candidates)]

    state_seq = -1 * torch.ones((ctc_probs.size(0), 1), dtype=torch.int16)

    candidates = torch.tensor([
        log_alpha[-1, len(y_insert_blank) - 1],
        log_alpha[-1, len(y_insert_blank) - 2]
    ])
    final_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
    state_seq[-1] = final_state[torch.argmax(candidates)]
    for t in range(ctc_probs.size(0) - 2, -1, -1):
        state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

    output_alignment = []
    for t in range(0, ctc_probs.size(0)):
        output_alignment.append(y_insert_blank[state_seq[t, 0]])

    return output_alignment
