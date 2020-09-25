# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Encoder definition."""
from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.encoder_layer import EncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling
from wenet.utils.mask import make_pad_mask


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size, eps=1e-12),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                PositionalEncoding(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size,
                                           dropout_rate)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        self.encoders = torch.nn.ModuleList([
            EncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            encoder output tensor, lens and mask
        """
        masks = ~make_pad_mask(ilens).unsqueeze(1)

        if isinstance(self.embed, Conv2dSubsampling):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        for layer in self.encoders:
            xs_pad, masks = layer(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, masks
