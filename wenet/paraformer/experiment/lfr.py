import math
from typing import Tuple
import torch


class LFR(torch.nn.Module):

    def __init__(self, m: int = 7, n: int = 6) -> None:
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.

        """
        super().__init__()

        self.m = m
        self.n = n

        self.left_padding_nums = math.ceil((self.m - 1) // 2)

    def forward(self, input: torch.Tensor,
                input_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, D = input.size()
        n_lfr = torch.ceil(input_lens / self.n)
        # print(n_lfr)
        # right_padding_nums >= 0
        prepad_nums = input_lens + self.left_padding_nums

        right_padding_nums = torch.where(
            self.m >= (prepad_nums - self.n * (n_lfr - 1)),
            self.m - (prepad_nums - self.n * (n_lfr - 1)),
            0,
        )
        T_all = self.left_padding_nums + input_lens + right_padding_nums

        new_len = T_all // self.n

        T_all_max = T_all.max().int()

        tail_frames_index = (input_lens - 1).view(B, 1, 1).repeat(1, 1,
                                                                  D)  # [B,1,D]

        tail_frames = torch.gather(input, 1, tail_frames_index)
        tail_frames = tail_frames.repeat(1, right_padding_nums.max().int(), 1)
        head_frames = input[:, 0:1, :].repeat(1, self.left_padding_nums, 1)

        # stack
        input = torch.cat([head_frames, input, tail_frames], dim=1)

        index = torch.arange(T_all_max,
                             device=input.device,
                             dtype=input_lens.dtype).unsqueeze(0).repeat(
                                 B, 1)  # [B, T_all_max]
        index_mask = (index <
                      (self.left_padding_nums + input_lens).unsqueeze(1)
                      )  #[B, T_all_max]

        tail_index_mask = torch.logical_not(
            index >= (T_all.unsqueeze(1))) & index_mask
        tail = torch.ones(T_all_max,
                          dtype=input_lens.dtype,
                          device=input.device).unsqueeze(0).repeat(B, 1) * (
                              T_all_max - 1)  # [B, T_all_max]
        indices = torch.where(torch.logical_or(index_mask, tail_index_mask),
                              index, tail)
        input = torch.gather(input, 1, indices.unsqueeze(2).repeat(1, 1, D))

        input = input.unfold(1, self.m, step=self.n).transpose(2, 3)
        # new len
        return input.reshape(B, -1, D * self.m), new_len
