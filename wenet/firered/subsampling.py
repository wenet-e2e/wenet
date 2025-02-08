# Copyright (c) 2025 Wenet Community. authors: Mddct(Dinghao Zhou)
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

from typing import Tuple, Union

import torch
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.utils.mask import make_non_pad_mask


class FireRedConv2dSubsampling4(Conv2dSubsampling4):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 idim: int,
                 d_model: int,
                 dropout_rate: float,
                 pos_enc_class: torch.nn.Module,
                 odim: int = 32):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__(idim, d_model, dropout_rate, pos_enc_class)
        del self.conv, self.out
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), d_model))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_lens = torch.sum(x_mask.squeeze(1), dim=1)
        x_lens = x_lens + self.right_context
        x_mask = make_non_pad_mask(x_lens).unsqueeze(1)
        x = torch.nn.functional.pad(x, (0, 0, 0, self.right_context),
                                    'constant', 0.0)
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return x, pos_emb, mask
