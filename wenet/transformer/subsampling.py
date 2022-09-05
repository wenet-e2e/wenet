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
# Modified from ESPnet(https://github.com/espnet/espnet)


"""Subsampling layer definition."""

from typing import Tuple, Union

import torch


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int
    ) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module
    ):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class TimeReduction2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        # NOTE(hslee): kernel_size is 3 in original paper.
        #   (https://arxiv.org/pdf/2206.00888.pdf)
        # Despite that, kernel_size of 2 is used for
        #   exact T -> ceil(T/2) time reduction without caching.
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, idim, 2, 2, groups=odim),  # depthwise conv
            torch.nn.Conv1d(idim, odim, 1, 1),  # pointwise conv
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            chunk_mask (torch.Tensor): Chunk mask (#batch, time, time).
            pos_emb (torch.Tensor): Positional embedding #(#batch, time, pe_dim)
        Returns:
            torch.Tensor: Subsampled tensor (#batch, odim, time'),
                where time' = ceil(time / 2).
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = ceil(time / 2).
            torch.Tensor: Subsampled mask (#batch, time', time'),
                where time' = ceil(time / 2).

        """
        x = x.transpose(1, 2)  # (batch, idim, time)
        x = torch.nn.functional.pad(x, (0, 1), "constant", 0.0)  # (batch, odim, time+1)
        x = self.conv(x)  # (batch, odim, time')
        x = x.transpose(1, 2)
        return x, x_mask[:, :, ::2], chunk_mask[:, ::2, ::2], pos_emb[:, ::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
        depthwise_separable: bool = False,
    ):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        if not depthwise_separable:
            conv2 = torch.nn.Conv2d(odim, odim, 3, 2)
        else:
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(odim, odim, 3, 2, groups=odim),  # depthwise conv
                torch.nn.Conv2d(odim, odim, 1, 1),  # pointwise conv
            )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            conv2,
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
        depthwise_separable: bool = False,
    ):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        if not depthwise_separable:
            conv2 = torch.nn.Conv2d(odim, odim, 5, 3)
        else:
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(odim, odim, 5, 3, groups=odim),  # depthwise conv
                torch.nn.Conv2d(odim, odim, 1, 1),  # pointwise conv
            )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            conv2,
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
        depthwise_separable: bool = False,
    ):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        assert (
            not depthwise_separable
        ), "depthwise separable conv for subsampling8 is not implemented."
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
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
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
