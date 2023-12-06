# Copyright (c) 2022 Ximalaya Inc. (authors: Yuguang Yang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conv2d Module with Valid Padding"""

import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd, _size_2_t, Union, _pair, Tensor, Optional


class Conv2dValid(_ConvNd):
    """
    Conv2d operator for VALID mode padding.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None,
            valid_trigx: bool = False,
            valid_trigy: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2dValid,
              self).__init__(in_channels, out_channels,
                             kernel_size_, stride_, padding_, dilation_, False,
                             _pair(0), groups, bias, padding_mode,
                             **factory_kwargs)
        self.valid_trigx = valid_trigx
        self.valid_trigy = valid_trigy

    def _conv_forward(self, input: Tensor, weight: Tensor,
                      bias: Optional[Tensor]):
        validx, validy = 0, 0
        if self.valid_trigx:
            validx = (input.size(-2) *
                      (self.stride[-2] - 1) - 1 + self.kernel_size[-2]) // 2
        if self.valid_trigy:
            validy = (input.size(-1) *
                      (self.stride[-1] - 1) - 1 + self.kernel_size[-1]) // 2
        return F.conv2d(input, weight, bias, self.stride, (validx, validy),
                        self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
