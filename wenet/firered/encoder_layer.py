from typing import Optional

import torch
from torch import nn
from wenet.transformer.encoder_layer import ConformerEncoderLayer


class FireRedConformerEncoderLayer(ConformerEncoderLayer):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: Optional[nn.Module] = None,
                 feed_forward_macaron: Optional[nn.Module] = None,
                 conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 layer_norm_type: str = 'layer_norm',
                 norm_eps: float = 0.00001):
        super().__init__(size, self_attn, feed_forward, feed_forward_macaron,
                         conv_module, dropout_rate, normalize_before,
                         layer_norm_type, norm_eps)
        del self.norm_mha
        self.norm_mha = torch.nn.Identity()
