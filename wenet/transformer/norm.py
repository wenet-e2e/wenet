import torch


class RMSNorm(torch.nn.Module):
    """ https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return x * (1 + self.weight)
        else:
            return x * self.weight
