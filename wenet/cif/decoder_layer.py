# Copyright (c) 2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License. Modified from
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DecoderLayerSANM(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear`
            instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first
                block.
    """

    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            src_attn: nn.Module,
            feed_forward: nn.Module,
            dropout_rate: float,
            normalize_before: bool = True,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayerSANM, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        if self_attn is not None:
            self.norm2 = nn.LayerNorm(size, eps=1e-12)
        if src_attn is not None:
            self.norm3 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(
            self,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: Optional[torch.Tensor] = None,
            cache: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
            Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in,
                size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # tgt = self.dropout(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            if self.training:
                cache = None
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x = residual + self.dropout(self.src_attn(x, memory, memory_mask))

        return x, tgt_mask, memory, memory_mask, cache
