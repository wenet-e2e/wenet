import torch
import torch.nn as nn


class ResidualModule(nn.Module):
    """
    Residual Connection Module for Squeezeformer Encoder Layer
    """
    def __init__(self, layer: nn.Module, coef: float = 1.0):
        super(ResidualModule, self).__init__()
        self.layer = layer
        self.coef = coef

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pos_emb: torch.Tensor,
                mask_pad: torch.Tensor = torch.ones(
                    (0, 0, 0), dtype=torch.bool),
                att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                ):
        x, mask, new_att_cache, new_cnn_cache = self.layer(
            x, mask, pos_emb, mask_pad, att_cache, cnn_cache)
        x = x * self.coef + x
        return x, mask, new_att_cache, new_cnn_cache
