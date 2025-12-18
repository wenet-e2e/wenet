# Copyright (c) 2022 Yifan Peng (Carnegie Mellon University)
#               2023 Voicecomm Inc (Kai Li)
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
"""MLP with convolutional gating (cgMLP) definition.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050

"""

from typing import Tuple
import torch
import torch.nn as nn
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
        causal: bool = True,
    ):
        super().__init__()

        # split input channels
        n_channels = size // 2
        self.norm = nn.LayerNorm(n_channels)
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            padding,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = WENET_ACTIVATION_CLASSES[gate_activation]()

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)
        # exchange the temporal dimension and the feature dimension
        x_g = x_g.transpose(1, 2)  # (#batch, channels, time)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x_g = nn.functional.pad(x_g, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x_g.size(0)  # equal batch
                assert cache.size(1) == x_g.size(1)  # equal channel
                x_g = torch.cat((cache, x_g), dim=2)
            assert (x_g.size(2) > self.lorder)
            new_cache = x_g[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0),
                                    dtype=x_g.dtype,
                                    device=x_g.device)

        x_g = x_g.transpose(1, 2)
        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out, new_cache


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
        causal: bool = True,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            causal=causal,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask. Not used yet
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """

        xs_pad = x

        # size -> linear_units
        xs_pad = self.channel_proj1(xs_pad)

        # linear_units -> linear_units/2
        xs_pad, new_cnn_cache = self.csgu(xs_pad, cache)

        # linear_units/2 -> size
        xs_pad = self.channel_proj2(xs_pad)

        out = xs_pad

        return out, new_cnn_cache
