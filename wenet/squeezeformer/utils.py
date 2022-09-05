import torch
import torch.nn as nn


class ResidualModule(nn.Module):
    def __init__(self, layer: nn.Module, coef: float = 1.0):
        super(ResidualModule, self).__init__()
        self.layer = layer
        self.coef = coef

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        return (self.layer(x, mask, pos_emb) * self.coef) + x


def recover_resolution(inputs: torch.Tensor) -> torch.Tensor:
    outputs = list()

    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
    return torch.stack(outputs, dim=1)
