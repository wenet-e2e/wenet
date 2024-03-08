# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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
"""Decoder self-attention layer definition."""
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from wenet.utils.class_utils import WENET_NORM_CLASSES


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
            If `None` is passed, Inter-attention is not used, such as
            CIF, GPT, and other decoder only model.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.norm1 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm2 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm3 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[Dict[str, Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        if cache is not None:
            att_cache = cache['self_att_cache']
            cross_att_cache = cache['cross_att_cache']
        else:
            att_cache, cross_att_cache = None, None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if att_cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
            att_cache = torch.empty(0, 0, 0, 0)
        else:
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        x, new_att_cache = self.self_attn(
            tgt_q,
            tgt_q,
            tgt_q,
            tgt_q_mask,
            cache=att_cache,
        )
        if cache is not None:
            cache['self_att_cache'] = new_att_cache
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)
            if cross_att_cache is None:
                cross_att_cache = torch.empty(0, 0, 0, 0)
            x, new_cross_cache = self.src_attn(x,
                                               memory,
                                               memory,
                                               memory_mask,
                                               cache=cross_att_cache)
            if cache is not None:
                cache['cross_att_cache'] = new_cross_cache
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask, memory, memory_mask
