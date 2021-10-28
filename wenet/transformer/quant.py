#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Simulate the quantize and dequantize operations definition."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair


class FakeQuantize(torch.quantization.FakeQuantize):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Args:
        symmetric: (Optional) If true, use symmetric quantization limits instead of
            training the minimum and maximum of each quantization range separately.
    """

    def __init__(self, symmetric=False):
        qscheme = torch.per_tensor_affine
        if symmetric:
            qscheme = torch.per_tensor_symmetric

        super().__init__(observer=torch.quantization.MovingAverageMinMaxObserver,
                         quant_min=-128, quant_max=127,
                         dtype=torch.qint8,
                         qscheme=qscheme,
                         reduce_range=False,
                         factory_kwargs=None)


class QuantLinear(nn.Linear):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        self._input_quantizer = FakeQuantize()
        self._weight_quantizer = FakeQuantize(symmetric=True)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output


class _QuantConvNd(torch.nn.modules.conv._ConvNd):
    """base class of quantized Conv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_QuantConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           transposed, output_padding, groups, bias, padding_mode)

        self._input_quantizer = FakeQuantize()
        self._weight_quantizer = FakeQuantize(symmetric=True)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)


class QuantConv2d(_QuantConvNd):
    """Quantized 2D conv"""

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

        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

        return output


class QuantConv1d(_QuantConvNd):
    """Quantized 1D Conv"""

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

        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _single(
                                              0), groups, bias, padding_mode)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv1d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _single(0), self.dilation, self.groups)
        else:
            output = F.conv1d(quant_input, quant_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output
