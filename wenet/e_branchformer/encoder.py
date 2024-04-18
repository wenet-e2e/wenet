# Copyright (c) 2022 Yifan Peng (Carnegie Mellon University)
#               2023 Voicecomm Inc (Kai Li)
#               2023 Lucky Wong
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
"""Encoder definition."""

import torch
from typing import List, Optional, Union
from wenet.branchformer.encoder import LayerDropModuleList

from wenet.e_branchformer.encoder_layer import EBranchformerEncoderLayer
from wenet.branchformer.cgmlp import ConvolutionalGatingMLP
from wenet.transformer.encoder import ConformerEncoder
from wenet.utils.class_utils import (
    WENET_ACTIVATION_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_MLP_CLASSES,
)


class EBranchformerEncoder(ConformerEncoder):
    """E-Branchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        selfattention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        activation_type: str = "swish",
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        causal: bool = False,
        merge_conv_kernel: int = 3,
        use_ffn: bool = True,
        macaron_style: bool = True,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super().__init__(input_size,
                         output_size,
                         attention_heads,
                         linear_units,
                         num_blocks,
                         dropout_rate,
                         positional_dropout_rate,
                         attention_dropout_rate,
                         input_layer,
                         pos_enc_layer_type,
                         True,
                         static_chunk_size,
                         use_dynamic_chunk,
                         global_cmvn,
                         use_dynamic_left_chunk,
                         1,
                         macaron_style,
                         selfattention_layer_type,
                         activation_type,
                         query_bias=query_bias,
                         key_bias=key_bias,
                         value_bias=value_bias,
                         conv_bias=conv_bias,
                         gradient_checkpointing=gradient_checkpointing,
                         use_sdpa=use_sdpa,
                         layer_norm_type=layer_norm_type,
                         norm_eps=norm_eps,
                         n_kv_head=n_kv_head,
                         head_dim=head_dim,
                         mlp_type=mlp_type,
                         mlp_bias=mlp_bias,
                         n_expert=n_expert,
                         n_expert_activated=n_expert_activated)

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
        )

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (output_size, cgmlp_linear_units, cgmlp_conv_kernel,
                            dropout_rate, use_linear_after_conv,
                            gate_activation, causal)

        # feed-forward module definition
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})")

        self.encoders = LayerDropModuleList(
            p=stochastic_depth_rate,
            modules=[
                EBranchformerEncoderLayer(
                    output_size,
                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args),
                    cgmlp_layer(*cgmlp_layer_args),
                    mlp_class(*positionwise_layer_args) if use_ffn else None,
                    mlp_class(*positionwise_layer_args)
                    if use_ffn and macaron_style else None,
                    dropout_rate,
                    merge_conv_kernel=merge_conv_kernel,
                    causal=causal,
                    stochastic_depth_rate=stochastic_depth_rate[lnum],
                ) for lnum in range(num_blocks)
            ])
