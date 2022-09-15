import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.subsampling import BaseSubsampling
from typing import Tuple, Union, Optional, Dict
from wenet.squeezeformer.conv2d import Conv2dValid


class DepthwiseConv2dSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Module): position encoding class.

        """

    def __init__(
            self, idim: int, odim: int, pos_enc_class: torch.nn.Module):
        super(DepthwiseConv2dSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = nn.Conv2d(in_channels=idim, out_channels=odim, kernel_size=3, stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(b, t, c * f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class TimeReductionLayer(nn.Module):
    def __init__(
            self, ichannel: int = 1, ochannel: int = 1,
            kernel_size: int = 5, stride: int = 2, encoder_dim: int = 256):
        super(TimeReductionLayer, self).__init__()
        self.ichannel = ichannel
        self.kernel_size = kernel_size
        self.dw_conv = Conv2dValid(
            in_channels=ichannel,
            out_channels=ochannel,
            kernel_size=(kernel_size, 1),
            stride=stride,
            valid_trigy=True
        )
        self.pw_conv = Conv2dValid(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=1,
            stride=1,
            valid_trigx=False,
            valid_trigy=False,
        )

        self.in_channels = ichannel
        self.kernel_size = kernel_size
        self.stride = stride
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.ichannel ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = xs.unsqueeze(2)
        padding1 = self.kernel_size - self.stride
        xs = F.pad(xs, (0, 0, 0, 0, 0, padding1, 0, 0), mode='constant', value=0.)
        xs = self.dw_conv(xs.transpose(1, 2))
        xs = xs.permute(0, 3, 1, 2).contiguous()
        xs = self.pw_conv(xs).permute(0, 2, 3, 1).squeeze(1).contiguous()
        tmp_length = xs.size(1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        padding2 = max(0, (xs_lens.max() - tmp_length).data.item())
        batch_size, hidden = xs.size(0), xs.size(-1)
        dummy_pad = torch.zeros(batch_size, padding2, hidden).to(xs.device)
        xs = torch.cat([xs, dummy_pad], dim=1)
        return xs, xs_lens
