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
        *args,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        **kwargs
    ):
        """Construct TransformerEncoder with LoRA parameters

        Args:
            *args: Arguments for the TransformerEncoder.
            **kwargs: Keyword arguments for the TransformerEncoder.
            lora_rank (int): Rank for LoRA.
            lora_alpha (int): Alpha for LoRA.
            lora_dropout (float): Dropout rate for LoRA.
            lora_list (Optional[List[str]]): List of layers to apply LoRA.
        """
        super().__init__(*args, **kwargs)
        activation = WENET_ACTIVATION_CLASSES[kwargs.get('activation_type',
                                                         'relu')]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            kwargs.get('attention_heads', 4),
            kwargs.get('output_size', 256),
            kwargs.get('attention_dropout_rate', 0.0),
            kwargs.get('query_bias', True),
            kwargs.get('key_bias', True),
            kwargs.get('value_bias', True),
            kwargs.get('use_sdpa', False),
            kwargs.get('n_kv_head', None),
            kwargs.get('head_dim', None),
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_list,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            kwargs.get('output_size', 256),
            kwargs.get('linear_units', 2048),
            kwargs.get('dropout_rate', 0.1),
            activation,
            kwargs.get('mlp_bias', True),
        )

        mlp_class = WENET_MLP_CLASSES[kwargs.get('mlp_type',
                                                 'position_wise_feed_forward')]
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                kwargs.get('output_size', 256),
                WENET_LORA_ATTENTION_CLASSES["selfattn"](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                kwargs.get('dropout_rate', 0.1),
                kwargs.get('normalize_before', True),
                layer_norm_type=kwargs.get('layer_norm_type', 'layer_norm'),
                norm_eps=kwargs.get('norm_eps', 1e-5),
            ) for _ in range(kwargs.get('num_blocks', 6))
        ])


class LoRAConformerEncoder(ConformerEncoder):
    """Conformer encoder module with lora."""

    def __init__(
        self,
        *args,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        **kwargs
    ):
        """Construct ConformerEncoder with LoRA parameters

        Args:
            *args: Arguments for the ConformerEncoder.
            **kwargs: Keyword arguments for the ConformerEncoder.
            lora_rank (int): Rank for LoRA.
            lora_alpha (int): Alpha for LoRA.
            lora_dropout (float): Dropout rate for LoRA.
            lora_list (Optional[List[str]]): List of layers to apply LoRA.
        """
        super().__init__(*args, **kwargs)
        activation = WENET_ACTIVATION_CLASSES[kwargs.get('activation_type',
                                                         'swish')]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            kwargs.get('attention_heads', 4),
            kwargs.get('output_size', 256),
            kwargs.get('attention_dropout_rate', 0.0),
            kwargs.get('query_bias', True),
            kwargs.get('key_bias', True),
            kwargs.get('value_bias', True),
            kwargs.get('use_sdpa', False),
            kwargs.get('n_kv_head', None),
            kwargs.get('head_dim', None),
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_list,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            kwargs.get('output_size', 256),
            kwargs.get('linear_units', 2048),
            kwargs.get('dropout_rate', 0.1),
            activation,
            kwargs.get('mlp_bias', True),
        )
        # convolution module definition
        convolution_layer_args = (
            kwargs.get('output_size', 256),
            kwargs.get('cnn_module_kernel', 15),
            activation,
            kwargs.get('cnn_module_norm', 'batch_norm'),
            kwargs.get('causal', False),
            kwargs.get('conv_bias', True)
        )

        mlp_class = WENET_MLP_CLASSES[kwargs.get('mlp_type',
                                                 'position_wise_feed_forward')]
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                kwargs.get('output_size', 256),
                WENET_LORA_ATTENTION_CLASSES[
                    kwargs.get('selfattention_layer_type', 'rel_selfattn')
                ](*encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args)
                    if kwargs.get('macaron_style', True) else None,
                ConvolutionModule(
                    *convolution_layer_args
                ) if kwargs.get('use_cnn_module', True) else None,
                kwargs.get('dropout_rate', 0.1),
                kwargs.get('normalize_before', True),
                layer_norm_type=kwargs.get('layer_norm_type', 'layer_norm'),
                norm_eps=kwargs.get('norm_eps', 1e-5),
            ) for _ in range(kwargs.get('num_blocks', 6))
        ])
