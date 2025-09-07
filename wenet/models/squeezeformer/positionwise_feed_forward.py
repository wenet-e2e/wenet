# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Ximalaya Inc (Yuguang Yang)
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
"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.idim = idim
        self.hidden_units = hidden_units
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.ada_scale = None
        self.ada_bias = None
        self.adaptive_scale = adaptive_scale
        self.ada_scale = torch.nn.Parameter(torch.ones([1, 1, idim]),
                                            requires_grad=adaptive_scale)
        self.ada_bias = torch.nn.Parameter(torch.zeros([1, 1, idim]),
                                           requires_grad=adaptive_scale)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        ffn1_max = self.idim**-0.5
        ffn2_max = self.hidden_units**-0.5
        torch.nn.init.uniform_(self.w_1.weight.data, -ffn1_max, ffn1_max)
        torch.nn.init.uniform_(self.w_1.bias.data, -ffn1_max, ffn1_max)
        torch.nn.init.uniform_(self.w_2.weight.data, -ffn2_max, ffn2_max)
        torch.nn.init.uniform_(self.w_2.bias.data, -ffn2_max, ffn2_max)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        if self.adaptive_scale:
            xs = self.ada_scale * xs + self.ada_bias
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
