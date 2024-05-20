import torch
import torch.nn as nn
from typing import List

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_dim if i == 0 else mid_dim // 2,
                mid_dim if i < self.n_layers - 1 else out_dim * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out.squeeze(-1)

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).contiguous()  # -> B x T x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

class ConvLinearBottleNeck(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        bottleneck_mid_dim: int,
        conv_kernel_sizes: List[int] = (3, 3)
    ):
        super(ConvLinearBottleNeck, self).__init__()

        self.subsampling = Conv1dSubsampler(encoder_dim, 2 * encoder_dim, decoder_dim, conv_kernel_sizes)

        self.activation = nn.GELU()
        self.fc1 = nn.Linear(decoder_dim, bottleneck_mid_dim, bias=False)
        self.fc2 = nn.Linear(bottleneck_mid_dim, decoder_dim, bias=False)

    def forward(self, x, x_lengths):
        x, out_lengths = self.subsampling(x, x_lengths)
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return residual + x, out_lengths