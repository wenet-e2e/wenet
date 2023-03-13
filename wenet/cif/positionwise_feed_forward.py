# Copyright (c) 2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License. Modified from
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

import torch


class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, adim=None,
                 activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim if adim is None else adim,
                                   bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = torch.nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))
