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

"""BranchformerEncoderLayer definition."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BranchformerEncoderLayer(torch.nn.Module):
    """Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention, optional
        cgmlp: ConvolutionalGatingMLP, optional
        dropout_rate (float): dropout probability
        merge_method (str): concat, learned_ave, fixed_ave
        cgmlp_weight (float): weight of the cgmlp branch, between 0 and 1,
            used if merge_method is fixed_ave
        attn_branch_drop_rate (float): probability of dropping the attn branch,
            used if merge_method is learned_ave
        stochastic_depth_rate (float): stochastic depth probability
    """

    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        cgmlp_weight: float = 0.5,
        attn_branch_drop_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        assert (attn is not None) or (
            cgmlp is not None
        ), "At least one branch should be valid"

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_two_branches = (attn is not None) and (cgmlp is not None)

        if attn is not None:
            self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = nn.LayerNorm(size)  # for the MLP module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        # # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(size, 1)
        self.pooling_proj2 = torch.nn.Linear(size, 1)

        # # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(size, 1)
        self.weight_proj2 = torch.nn.Linear(size, 1)

        if self.use_two_branches:
            if self.merge_method == "concat":
                self.merge_proj = torch.nn.Linear(size + size, size)

            elif self.merge_method == "learned_ave":
                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            elif self.merge_method == "fixed_ave":
                assert (
                    0.0 <= cgmlp_weight <= 1.0
                ), "cgmlp weight should be between 0.0 and 1.0"

                # remove the other branch if only one branch is used
                if cgmlp_weight == 0.0:
                    self.use_two_branches = False
                    self.cgmlp = None
                    self.norm_mlp = None
                elif cgmlp_weight == 1.0:
                    self.use_two_branches = False
                    self.attn = None
                    self.norm_mha = None

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)
            else:
                raise ValueError(f"unknown merge method: {merge_method}")
        else:
            self.merge_proj = torch.nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        stoch_layer_coeff = 1.0
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            return x, mask, att_cache, cnn_cache

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
            x1 = self.norm_mha(x1)
            x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
            x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)
            x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
            x2 = self.dropout(x2)

        # Merge two branches
        if self.use_two_branches:
            if self.merge_method == "concat":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(torch.cat([x1, x2], dim=-1))
                )
            elif self.merge_method == "learned_ave":
                if (
                    self.training
                    and self.attn_branch_drop_rate > 0
                    and torch.rand(1).item() < self.attn_branch_drop_rate
                ):
                    # Drop the attn branch
                    w1, w2 = torch.tensor(0.0), torch.tensor(1.0)
                else:
                    # branch1
                    score1 = (self.pooling_proj1(x1).transpose(1, 2) / self.size**0.5)
                    score1 = score1.masked_fill(mask_pad.eq(0), -float('inf'))
                    score1 = torch.softmax(score1, dim=-1).masked_fill(
                        mask_pad.eq(0), 0.0
                    )

                    pooled1 = torch.matmul(score1, x1).squeeze(1)  # (batch, size)
                    weight1 = self.weight_proj1(pooled1)  # (batch, 1)

                    # branch2
                    score2 = (self.pooling_proj2(x2).transpose(1, 2) / self.size**0.5)
                    score2 = score2.masked_fill(mask_pad.eq(0), -float('inf'))
                    score2 = torch.softmax(score2, dim=-1).masked_fill(
                        mask_pad.eq(0), 0.0
                    )

                    pooled2 = torch.matmul(score2, x2).squeeze(1)  # (batch, size)
                    weight2 = self.weight_proj2(pooled2)  # (batch, 1)

                    # normalize weights of two branches
                    merge_weights = torch.softmax(
                        torch.cat([weight1, weight2], dim=-1), dim=-1
                    )  # (batch, 2)
                    merge_weights = merge_weights.unsqueeze(-1).unsqueeze(
                        -1
                    )  # (batch, 2, 1, 1)
                    w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(w1 * x1 + w2 * x2)
                )
            elif self.merge_method == "fixed_ave":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(
                        (1.0 - self.cgmlp_weight) * x1 + self.cgmlp_weight * x2
                    )
                )
            else:
                raise RuntimeError(f"unknown merge method: {self.merge_method}")
        else:
            if self.attn is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x2))
            elif self.cgmlp is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x1))
            else:
                # This should not happen
                raise RuntimeError("Both branches are not None, which is unexpected.")

        x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
