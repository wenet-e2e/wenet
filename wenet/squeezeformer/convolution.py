# Copyright (c) 2022, Sangchun Ha. All rights reserved.
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
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional, Dict

from wenet.transformer.activations import Swish, GLU


class PointwiseConv1d(torch.nn.Module):
    def __init__(self, ichannel: int, ochannel: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super(PointwiseConv1d, self).__init__()
        self.pw_conv = nn.Conv1d(
            in_channels=ichannel,
            out_channels=ochannel,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw_conv(x)


class DepthwiseConv1d(nn.Module):
    def __init__(self, ichannel: int, ochannel: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(DepthwiseConv1d, self).__init__()
        assert ochannel % ichannel == 0
        self.dw_conv = nn.Conv1d(
            in_channels=ichannel,
            out_channels=ochannel,
            kernel_size=kernel_size,
            groups=ichannel,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)


class DepthwiseConv2d(nn.Module):
    def __init__(
            self,
            ichannel: int,
            ochannel: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: Union[int, str] = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert ochannel % ichannel == 0
        self.dw_conv = nn.Conv2d(
            in_channels=ichannel,
            out_channels=ochannel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=ichannel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)


class ConvolutionModule(torch.nn.Module):
    def __init__(self, ichannel: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_rate: float = 0.1):
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2
        self.pw_conv1 = PointwiseConv1d(ichannel=ichannel, ochannel=ichannel * expansion_factor,
                                        stride=1, padding=0, bias=True)
        self.glu = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(ichannel=ichannel, ochannel=ichannel, kernel_size=kernel_size,
                                       stride=1, padding=(kernel_size - 1) // 2)
        self.norm = nn.BatchNorm1d(ichannel)
        self.activation = Swish()
        self.pw_conv2 = PointwiseConv1d(ichannel=ichannel, ochannel=ichannel, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        x = self.dropout(x).transpose(1, 2)
        return x


if __name__ == '__main__':
    # module = PointwiseConv1d(1, 256)
    module = ConvolutionModule(ichannel=256)
    # x = torch.rand(1, 1, 128)
    x = torch.rand(2, 31, 256)
    y = module(x)
    print('y', y.size())
