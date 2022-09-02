import torch
import torch.nn as nn


class ResidualModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0):
        super(ResidualModule, self).__init__()
        self.module = module
        self.factor = module_factor

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return (self.module(x, *args) * self.factor) + x


def recover_resolution(inputs: torch.Tensor) -> torch.Tensor:
    outputs = list()

    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
    return torch.stack(outputs, dim=1)
