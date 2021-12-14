#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Simulate the quantize and dequantize operations definition."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair


class QuantLinear(nn.Linear):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.linear(x, self.weight, bias=self.bias)
        x = self.dequant(x)
        return x


class QuantConv2d(nn.Conv2d):
    """Quantized 2D conv

    Raises:
        ValueError: If unsupported arguments are passed in.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, bias, padding_mode)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.conv2d(F.pad(x, expanded_padding, mode='circular'),
                         self.weight, self.bias, self.stride,
                         _pair(0), self.dilation, self.groups)
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation,
                         self.groups)

        x = self.dequant(x)

        return x


class QuantConv1d(nn.Conv1d):
    """Quantized 1D Conv

    Raises:
        ValueError: If unsupported arguments are passed in.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                          groups, bias, padding_mode)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.conv1d(F.pad(x, expanded_padding, mode='circular'),
                         self.weight, self.bias, self.stride,
                         _single(0), self.dilation, self.groups)
        else:
            x = F.conv1d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        x = self.dequant(x)
        return x
