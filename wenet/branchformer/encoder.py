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
"""Encoder definition."""

import torch

from typing import List, Optional, Union

from wenet.branchformer.encoder_layer import BranchformerEncoderLayer
from wenet.branchformer.cgmlp import ConvolutionalGatingMLP
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.class_utils import (
    WENET_ATTENTION_CLASSES, )


class BranchformerEncoder(BaseEncoder):
    """Branchformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        use_attn: bool = True,
        attention_heads: int = 4,
        selfattention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        use_cgmlp: bool = True,
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        merge_method: str = "concat",
        cgmlp_weight: Union[float, List[float]] = 0.5,
        attn_branch_drop_rate: Union[float, List[float]] = 0.0,
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
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__(input_size, output_size, attention_heads,
                         cgmlp_linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, True,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing,
                         use_sdpa, layer_norm_type, norm_eps)

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
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
            causal,
        )

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})")

        if isinstance(cgmlp_weight, float):
            cgmlp_weight = [cgmlp_weight] * num_blocks
        if len(cgmlp_weight) != num_blocks:
            raise ValueError(
                f"Length of cgmlp_weight ({len(cgmlp_weight)}) should be equal to "
                f"num_blocks ({num_blocks})")

        if isinstance(attn_branch_drop_rate, float):
            attn_branch_drop_rate = [attn_branch_drop_rate] * num_blocks
        if len(attn_branch_drop_rate) != num_blocks:
            raise ValueError(
                f"Length of attn_branch_drop_rate ({len(attn_branch_drop_rate)}) "
                f"should be equal to num_blocks ({num_blocks})")

        self.encoders = LayerDropModuleList(
            p=stochastic_depth_rate,
            modules=[
                BranchformerEncoderLayer(
                    output_size,
                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args) if use_attn else None,
                    cgmlp_layer(*cgmlp_layer_args) if use_cgmlp else None,
                    dropout_rate,
                    merge_method,
                    cgmlp_weight[lnum],
                    attn_branch_drop_rate[lnum],
                    stochastic_depth_rate[lnum],
                ) for lnum in range(num_blocks)
            ])


# modify from : https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_drop.py # noqa
class LayerDropModuleList(torch.nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add

    Limitations:
        1 can work with ddp when layer's gradient checkpoint disabled
        2 can't work with ddp when layer's gradient checkpoint enables
        3 can work with fsdp
        4 can work with deepspeed
    """

    def __init__(self, p: List[float], modules=None):
        super().__init__(modules)
        assert len(p) == len(self)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p[i]):
                yield m
