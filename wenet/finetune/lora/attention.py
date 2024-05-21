# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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
"""Multi-Head Attention layer definition with lora."""

from typing import Optional, List

import torch
from torch import nn

from wenet.transformer.attention import (MultiHeadedAttention,
                                         RelPositionMultiHeadedAttention)
import wenet.finetune.lora.layers as lora


class LoRAMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with lora.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 8,
                 lora_dropout: float = 0.0,
                 lora_list: Optional[List[str]] = None):
        """Construct an MultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = lora.Linear(
            n_feat,
            n_feat,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        ) if lora_list and "o" in lora_list else nn.Linear(n_feat, n_feat)

        lora_qkv_dict = {
            "q": lora_list and "q" in lora_list,
            "k": lora_list and "k" in lora_list,
            "v": lora_list and "v" in lora_list
        }
        bias_dict = {"q": query_bias, "k": key_bias, "v": value_bias}

        for key, value in lora_qkv_dict.items():
            setattr(
                self, f"linear_{key}",
                lora.Linear(n_feat,
                            n_feat,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            bias=bias_dict[key]) if value else nn.Linear(
                                n_feat, n_feat, bias_dict[key]))
        self.dropout = nn.Dropout(p=dropout_rate)


class LoRARelPositionMultiHeadedAttention(LoRAMultiHeadedAttention,
                                          RelPositionMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 8,
                 lora_dropout: float = 0.0,
                 lora_list: Optional[List[str]] = None):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, lora_rank, lora_alpha,
                         lora_dropout, lora_list)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
