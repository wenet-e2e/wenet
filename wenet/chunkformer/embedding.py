"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch

class RelPositionalEncodingWithRightContext(torch.nn.Module):
    """Relative positional encoding module.

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncodingWithRightContext, self).__init__()

        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.xscale = math.sqrt(self.d_model)
        self.max_len = max_len
        self.extend_pe(max_len)

    def extend_pe(self, size: int) -> None:
        """Reset the positional encodings."""
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(size, self.d_model)
        pe_negative = torch.zeros(size, self.d_model)
        position = torch.arange(0, size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a
        # Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def position_encoding(
        self,
        chunk_size: Union[int, torch.Tensor] = 0,
        left_context_size: Union[int, torch.Tensor] = 0,
        right_context_size: Union[int, torch.Tensor] = 0,
        apply_dropout: bool = False,

    ) -> torch.Tensor:

        if isinstance(left_context_size, int):
            assert left_context_size + chunk_size < self.max_len
            x_size_1 = chunk_size + left_context_size
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + chunk_size + right_context_size,
            ]
        else:
            assert left_context_size + chunk_size < self.max_len
            x_size_1 = chunk_size + left_context_size
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + chunk_size + right_context_size,
            ]

        return pos_emb

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: Union[int, torch.Tensor] = 0,
        left_context_size: Union[int, torch.Tensor] = 0,
        right_context_size: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            offset (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.
            chunk_size (int): Chunk size for limited chunk context
            left_context_size (int): Left context size for limited chunk context
            right_context_size (int): Right context size for limited chunk context
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        x = x * self.xscale
        chunk_size = x.size(1) if chunk_size <= 0 else chunk_size
        pos_emb = self.position_encoding(
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            apply_dropout=False
        ).to(device=x.device, dtype=x.dtype)
        return self.dropout(x), self.dropout(pos_emb)
