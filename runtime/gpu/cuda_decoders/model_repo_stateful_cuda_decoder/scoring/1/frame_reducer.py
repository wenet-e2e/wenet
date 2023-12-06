#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corp.        (authors: Yifan   Yang,
#                                                    Zengwei Yao,
#                                                    Wei     Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.
    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


class FrameReducer(nn.Module):
    """The encoder output is first used to calculate
    the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some thresholds,
    it will be simply discarded from the encoder output.
    """

    def __init__(
        self,
        blank_threshlod: float = 0.95,
    ):
        super().__init__()
        self.blank_threshlod = blank_threshlod

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        ctc_output: torch.Tensor,
        y_lens: Optional[torch.Tensor] = None,
        blank_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
              The shared encoder output with shape [N, T, C].
            x_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `x` before padding.
            ctc_output:
              The CTC output with shape [N, T, vocab_size].
            y_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `y` before padding.
            blank_id:
              The blank id of ctc_output.
        Returns:
            out:
              The frame reduced encoder output with shape [N, T', C].
            out_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `out` before padding.
        """
        N, T, C = x.size()

        padding_mask = make_pad_mask(x_lens, x.size(1))
        non_blank_mask = (ctc_output[:, :, blank_id] < math.log(
            self.blank_threshlod)) * (~padding_mask)  # noqa

        if y_lens is not None:
            # Limit the maximum number of reduced frames
            limit_lens = T - y_lens
            max_limit_len = limit_lens.max().int()
            fake_limit_indexes = torch.topk(ctc_output[:, :, blank_id],
                                            max_limit_len).indices
            T = (torch.arange(max_limit_len).expand_as(
                fake_limit_indexes, ).to(device=x.device))
            T = torch.remainder(T, limit_lens.unsqueeze(1))
            limit_indexes = torch.gather(fake_limit_indexes, 1, T)
            limit_mask = torch.full_like(
                non_blank_mask,
                False,
                device=x.device,
            ).scatter_(1, limit_indexes, True)

            non_blank_mask = non_blank_mask | ~limit_mask

        out_lens = non_blank_mask.sum(dim=1)
        max_len = out_lens.max()
        pad_lens_list = (torch.full_like(
            out_lens,
            max_len.item(),
            device=x.device,
        ) - out_lens)
        max_pad_len = pad_lens_list.max()

        out = F.pad(x, (0, 0, 0, max_pad_len))

        valid_pad_mask = ~make_pad_mask(pad_lens_list)
        total_valid_mask = torch.concat([non_blank_mask, valid_pad_mask],
                                        dim=1)

        out = out[total_valid_mask].reshape(N, -1, C)

        return out, out_lens


if __name__ == "__main__":
    import time

    test_times = 1
    device = "cuda:0"
    frame_reducer = FrameReducer(1.0)

    # non zero case
    seq_len = 498
    x = torch.ones(15, seq_len, 384, dtype=torch.float32, device=device)
    x_lens = torch.tensor([seq_len] * 15, dtype=torch.int64, device=device)
    y_lens = torch.tensor([150] * 15, dtype=torch.int64, device=device)
    ctc_output = torch.log(
        torch.randn(15, seq_len, 500, dtype=torch.float32, device=device), )

    avg_time = 0
    for i in range(test_times):
        torch.cuda.synchronize(device=x.device)
        delta_time = time.time()
        x_fr, x_lens_fr = frame_reducer(x, x_lens, ctc_output)
        torch.cuda.synchronize(device=x.device)
        delta_time = time.time() - delta_time
        avg_time += delta_time
    print(x_fr.shape)
    print(x_lens_fr)
    print(avg_time / test_times)

    # all zero case
    x = torch.zeros(15, 498, 384, dtype=torch.float32, device=device)
    x_lens = torch.tensor([498] * 15, dtype=torch.int64, device=device)
    y_lens = torch.tensor([150] * 15, dtype=torch.int64, device=device)
    ctc_output = torch.zeros(15, 498, 500, dtype=torch.float32, device=device)

    avg_time = 0
    for i in range(test_times):
        torch.cuda.synchronize(device=x.device)
        delta_time = time.time()
        x_fr, x_lens_fr = frame_reducer(x, x_lens, ctc_output, y_lens)
        torch.cuda.synchronize(device=x.device)
        delta_time = time.time() - delta_time
        avg_time += delta_time
    print(x_fr.shape)
    print(x_lens_fr)
    print(avg_time / test_times)
