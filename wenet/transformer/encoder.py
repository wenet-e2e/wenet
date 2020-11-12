#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Encoder definition."""
from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.linear import InputLinear
from wenet.transformer.subsampling import Conv2dSubsampling
from wenet.transformer.subsampling import RelConv2dSubsampling
from wenet.transformer.subsampling import RelConv2dSubsampling6
from wenet.transformer.subsampling import RelConv2dSubsampling8
from wenet.utils.mask import make_pad_mask

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from wenet.transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
)
from wenet.utils.common import get_activation
import logging


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size, eps=1e-12),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                PositionalEncoding(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size,
                                           dropout_rate)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            encoder output tensor, lens and mask
        """
        masks = ~make_pad_mask(ilens).unsqueeze(1)

        if isinstance(self.embed, Conv2dSubsampling):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        for layer in self.encoders:
            xs_pad, masks = layer(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        return xs_pad, masks


class ConformerEncoder(torch.nn.Module):
    """Conformer encoder module.
    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: Optional[str] = "linear",
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        pos_enc_layer_type: Optional[str] = "rel_pos",
        selfattention_layer_type: Optional[str] = "rel_selfattn",
        activation_type: Optional[str] = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
    ):
        """Construct an Encoder object."""
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = InputLinear(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = RelConv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = RelConv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = RelConv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " +
                             selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        masks = ~make_pad_mask(ilens).unsqueeze(1)

        xs, masks = self.embed(xs, masks)

        for layer in self.encoders:
            xs, masks = layer(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
