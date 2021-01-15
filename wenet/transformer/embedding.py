#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Positonal Encoding Module."""

import math
from typing import Tuple

import torch


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        This func is not used in vanilla transformer,for the pos_emb
        is added in the input and will not calculated seperately.

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert size < self.max_len
        return self.dropout(self.pe[:size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, 2*time-1, `*`).
        """
        assert offset + x.size(1) < int(self.max_len / 2)
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        mid_pos = int(self.max_len / 2)
        L = x.size(1)
        pos_emb = self.pe[mid_pos - L + 1:mid_pos + L]
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        In chunk-based streaming mode, the forward function receives
        chunk inputs and only returns relative positional embeddings
        with length of the chunk input. This func is used to get
        relative positional embeddings for
        [-(cache_size+chunk_size), cache_size+chunk_size].

        See attention.py:RelPositionMultiHeadedAttention for more details

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Return
        Args:
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert size < int(self.max_len / 2)
        mid_pos = int(self.max_len / 2)
        pos_emb = self.pe[mid_pos - size + 1:mid_pos + size]
        return self.dropout(pos_emb)
