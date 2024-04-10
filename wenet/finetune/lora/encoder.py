# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alan (alanfangemail@gmail.com)
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
"""Encoder definition with lora."""

from typing import Optional, List

import torch

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.class_utils import (
    WENET_MLP_CLASSES,
    WENET_ACTIVATION_CLASSES,
)
from wenet.finetune.lora.utils import WENET_LORA_ATTENTION_CLASSES


class LoRATransformerEncoder(TransformerEncoder):
    """Transformer encoder module with lora."""

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
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        mlp_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        mlp_type: str = 'position_wise_feed_forward',
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, query_bias, key_bias,
                         value_bias, mlp_bias, activation_type,
                         gradient_checkpointing, use_sdpa, mlp_type,
                         layer_norm_type, norm_eps, n_kv_head, head_dim)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES["selfattn"](
                    attention_heads, output_size, attention_dropout_rate,
                    query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                    head_dim, lora_rank, lora_alpha, lora_dropout, lora_list),
                mlp_class(output_size, linear_units, dropout_rate, activation,
                          mlp_bias),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])


class LoRAConformerEncoder(ConformerEncoder):
    """Conformer encoder module with lora."""

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
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        mlp_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        mlp_type: str = 'position_wise_feed_forward',
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__(
            input_size, output_size, attention_heads, linear_units, num_blocks,
            dropout_rate, positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            static_chunk_size, use_dynamic_chunk, global_cmvn,
            use_dynamic_left_chunk, positionwise_conv_kernel_size,
            macaron_style, selfattention_layer_type, activation_type,
            use_cnn_module, cnn_module_kernel, causal, cnn_module_norm,
            query_bias, key_bias, value_bias, mlp_bias, conv_bias,
            gradient_checkpointing, use_sdpa, mlp_type, layer_norm_type,
            norm_eps, n_kv_head, head_dim)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_list,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, conv_bias)

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])
