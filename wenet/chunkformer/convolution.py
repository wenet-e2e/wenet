# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

"""ConvolutionModule definition."""

from typing import Tuple

import torch
from torch import nn

class ChunkConvolutionModule(nn.Module):
    """ConvolutionModule in ChunkFormer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True,
                 dynamic_conv: bool = False):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()
        self.dynamic_conv = dynamic_conv
        self.channels = channels
        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        elif dynamic_conv:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = 0
            self.lorder = (kernel_size - 1) // 2
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
        chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
            chunk_size (int): Chunk size for dynamic chunk convolution.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        if self.dynamic_conv and chunk_size <= 0:
            chunk_size = x.size(2)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad.to(torch.bool), 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        n_frames_pad = -1
        n_chunks = -1
        if self.dynamic_conv:
            size = self.lorder + chunk_size
            step = chunk_size

            n_frames_pad = (step - ((x.size(2) - size) % step)) % step
            # (batch, 2*channel, dim + n_frames_pad)
            x = torch.nn.functional.pad(x, (0, n_frames_pad))

            n_chunks = ((x.size(2) - size) // step) + 1
            # [B, C, n_chunks, size]
            x = x.unfold(-1, size=size, step=step)
            # [B, n_chunks, C, size]
            x = x.transpose(1, 2)
            # [B * n_chunks, C, size]
            x = x.reshape(-1, x.size(2), x.size(3))

            # pad right for dynamic conv
            x = nn.functional.pad(x, (0, self.lorder), 'constant', 0.0)


        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        if self.dynamic_conv:
            # [B, n_chunk, C, chunk_size]
            x = x.reshape(-1, n_chunks, x.size(1), x.size(2))
            # [B, C, n_chunks, chunk_size]
            x = x.transpose(1, 2)
            # [B, C, n_chunks * chunk_size]
            x = x.reshape(x.size(0), x.size(1), -1)
            # remove padding
            x = x[..., :x.size(2) - n_frames_pad]

        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad.to(torch.bool), 0.0)
        return x.transpose(1, 2), new_cache



    def forward_parallel_chunk(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0)),
        truncated_context_size: int = 0

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (channels, cache_t),
                (0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
        lorder = self.kernel_size // 2
        chunk_size = x.shape[-1]
        cache_t = cache.size(-1)
        if cache_t == 0:
            cache = torch.zeros(self.channels, lorder).to(x.device)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # ----------Overlapping Chunk Transformation-----------------------------------
        x = x.transpose(0, 1).reshape(self.channels, -1)  # [C, n_chunk * T]
        x = torch.cat([cache, x], dim=-1)

        # Streaming long-form transcription is disabled if input cache is empty
        if cache_t > 0:
            new_cache = x[:, :truncated_context_size + cache.size(-1)]
            new_cache = new_cache[:, -cache.size(-1):]
        else:
            new_cache = torch.zeros((0, 0))

        x = nn.functional.pad(x, (0, lorder), 'constant', 0.0)
        x = x.unfold(-1, chunk_size + 2 * lorder, chunk_size).transpose(0, 1)
        # [n_chunk +1, C, chunk_size + 2 * lorder]
        # -----------------------------------------------------------------------------

        if mask_pad.size(2) > 0:  # time > 0
            x = torch.where(mask_pad, x, 0)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad[:, :, lorder:-lorder], 0.0)

        return x.transpose(1, 2), new_cache
