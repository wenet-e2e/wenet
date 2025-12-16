# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)
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
"""Subsampling layer definition."""

from typing import Tuple, Union

import torch

from wenet.models.transformer.subsampling import BaseSubsampling


class AUTConv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length). for qwen3omni aut

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module, hidden_size: int = None):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        hidden_size = odim if hidden_size is None else hidden_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_size, 3, 2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_size, hidden_size, 3, 2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_size, hidden_size, 3, 2, padding=1),
            torch.nn.GELU(),
        )
        self.linear = torch.nn.Linear(
            hidden_size * ((((idim - 1 + 2 * 1) // 2 - 1 + 2 * 1) // 2 - 1 + 2 * 1) // 2),
            odim,
            bias=False)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1).transpose(2, 3)  # (b, c, t, f)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = self.linear(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 0::2][:, :, 0::2][:, :, 0::2]
