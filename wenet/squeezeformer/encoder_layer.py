# Copyright (c) 2022 Ximalaya Inc. (authors: Yuguang Yang)
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
"""SqueezeformerEncoderLayer definition."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SqueezeformerEncoderLayer(nn.Module):
    """Encoder layer module.
        Args:
            size (int): Input dimension.
            self_attn (torch.nn.Module): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward1 (torch.nn.Module): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (torch.nn.Module): Convolution module instance.
                `ConvlutionModule` instance can be used as the argument.
            feed_forward2 (torch.nn.Module): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward1: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        feed_forward2: Optional[nn.Module] = None,
        normalize_before: bool = False,
        dropout_rate: float = 0.1,
        concat_after: bool = False,
    ):
        super(SqueezeformerEncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.layer_norm1 = nn.LayerNorm(size)
        self.ffn1 = feed_forward1
        self.layer_norm2 = nn.LayerNorm(size)
        self.conv_module = conv_module
        self.layer_norm3 = nn.LayerNorm(size)
        self.ffn2 = feed_forward2
        self.layer_norm4 = nn.LayerNorm(size)
        self.normalize_before = normalize_before
        self.dropout = nn.Dropout(dropout_rate)
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # self attention module
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.layer_norm1(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm2(x)
        x = self.ffn1(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm2(x)

        # conv module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        residual = x
        if self.normalize_before:
            x = self.layer_norm3(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm3(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm4(x)
        x = self.ffn2(x)
        # we do not use dropout here since it is inside feed forward function
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm4(x)

        return x, mask, new_att_cache, new_cnn_cache
