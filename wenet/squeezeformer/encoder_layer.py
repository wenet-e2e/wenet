import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Dict
from wenet.squeezeformer.utils import ResidualModule


class SqueezeformerEncoderLayer(nn.Module):
    """Encoder layer module.
        Args:
            size (int): Input dimension.
            self_attn (torch.nn.Module): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward (torch.nn.Module): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (torch.nn.Module): Convolution module instance.
                `ConvlutionModule` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        """

    def __init__(
            self,
            size: int,
            self_attn: torch.nn.Module,
            feed_forward: Optional[nn.Module] = None,
            conv_module: Optional[nn.Module] = None,
            normalize_before: bool = False,
    ):
        super(SqueezeformerEncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.layer_norm1 = nn.LayerNorm(size)
        self.ffn1 = feed_forward
        self.layer_norm2 = nn.LayerNorm(size)
        self.conv_module = conv_module
        self.layer_norm3 = nn.LayerNorm(size)
        self.ffn2 = feed_forward
        self.layer_norm4 = nn.LayerNorm(size)
        self.normalize_before = normalize_before

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # self attention module
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)
        x = self.self_attn(x, x, x, mask, pos_emb)
        if not self.normalize_before:
            x = self.layer_norm1(x)
        x = residual + x

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm2(x)
        x = self.ffn1(x)
        if not self.normalize_before:
            x = self.layer_norm2(x)
        x = residual + x

        # conv module
        residual = x
        if self.normalize_before:
            x = self.layer_norm3(x)
        x = self.conv_module(x)
        if not self.normalize_before:
            x = self.layer_norm3(x)
        x = residual + x

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm4(x)
        x = self.ffn2(x)
        if not self.normalize_before:
            x = self.layer_norm4(x)
        x = residual + x
        return x


if __name__ == '__main__':
    from wenet.squeezeformer.convolution import ConvolutionModule
    from wenet.squeezeformer.ffn import PositionwiseFeedForward
    from wenet.squeezeformer.attention import RelPositionMultiHeadedAttention
    from wenet.squeezeformer.embedding import RelPositionalEncoding
    from wenet.utils.mask import make_pad_mask

    self_attn = RelPositionMultiHeadedAttention(4, 256, 0.1)
    ffn = PositionwiseFeedForward(256, 4, 0.1)
    conv_module = ConvolutionModule(256, 31, 2, 0.1)
    block = SqueezeformerEncoderLayer(256, self_attn, ffn, conv_module)
    x = torch.rand(2, 128, 256)
    mask = ~make_pad_mask(torch.tensor([128, 128])).unsqueeze(1)
    pos_emb = RelPositionalEncoding(256, 0.1, 5000)
    pos = pos_emb.position_encoding(0, 128)
    print('pos', pos.size())

    x = block(x, mask, pos)
    print('x', x.size())
    torch.jit.script(block)