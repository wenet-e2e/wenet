# Copyright (c) 2022 Yifan Peng (Carnegie Mellon University)
#               2023 Voicecomm Inc (Kai Li)
#               2023 Lucky Wong
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""EBranchformerEncoderLayer definition."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from wenet.transformer.attention import T_CACHE


class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
        self,
        size: int,
        attn: torch.nn.Module,
        cgmlp: torch.nn.Module,
        feed_forward: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_conv_kernel: int = 3,
        causal: bool = True,
        stochastic_depth_rate=0.0,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = nn.LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size)

        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        # for the final output of the block
        self.norm_final = nn.LayerNorm(size)

        self.dropout = torch.nn.Dropout(dropout_rate)

        if causal:
            padding = 0
            self.lorder = merge_conv_kernel - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (merge_conv_kernel - 1) % 2 == 0
            padding = (merge_conv_kernel - 1) // 2
            self.lorder = 0
        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=padding,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def _forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        stoch_layer_coeff: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)
        x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x2 = self.norm_mlp(x2)
        x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        if self.lorder > 0:
            x_tmp = nn.functional.pad(x_tmp, (self.lorder, 0), "constant", 0.0)
            assert x_tmp.size(2) > self.lorder
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + stoch_layer_coeff * self.dropout(
            self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward(x))

        x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        stoch_layer_coeff = 1.0
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training:
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        return self._forward(x, mask, pos_emb, mask_pad, att_cache, cnn_cache,
                             stoch_layer_coeff)
