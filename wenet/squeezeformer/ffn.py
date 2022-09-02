import torch
import torch.nn as nn
from wenet.transformer.activations import Swish


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int = 512, expansion_factor: int = 4, dropout_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(idim, idim * expansion_factor, bias=True)
        self.activation = Swish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.w2 = nn.Linear(idim * expansion_factor, idim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x
