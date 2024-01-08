from typing import Tuple, Union
import torch
from wenet.transformer.subsampling import BaseSubsampling


class IdentitySubsampling(BaseSubsampling):
    """ Paraformer subsampling
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        super().__init__()
        _, _ = idim, odim
        self.right_context = 6
        self.subsampling_rate = 6
        self.pos_enc = pos_enc_class

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[torch.Tensor, int] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time
            torch.Tensor: positional encoding

        """
        # NOTE(Mddct): Paraformer starts from 1
        if isinstance(offset, torch.Tensor):
            offset = torch.add(offset, 1)
        else:
            offset = offset + 1
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset + 1, size)
