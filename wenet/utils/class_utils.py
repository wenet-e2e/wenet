#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch

from wenet.transformer.subsampling import (
    LinearNoSubsampling, EmbedinigNoSubsampling,
    Conv1dSubsampling2, Conv2dSubsampling4,
    Conv2dSubsampling6, Conv2dSubsampling8,
)
from wenet.efficient_conformer.subsampling import Conv2dSubsampling2
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4
from wenet.transformer.embedding import (
    PositionalEncoding, RelPositionalEncoding,
    WhisperPositionalEncoding, LearnablePositionalEncoding,
    NoPositionalEncoding
)
from wenet.transformer.attention import (
    MultiHeadedAttention, RelPositionMultiHeadedAttention
)


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from wenet.transformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": getattr(torch.nn, "SiLU", Swish),
        "gelu": torch.nn.GELU
    }

    return activation_funcs[act]()


def get_rnn(rnn_type: str) -> torch.nn.Module:
    assert rnn_type in ["rnn", "lstm", "gru"]
    if rnn_type == "rnn":
        return torch.nn.RNN
    elif rnn_type == "lstm":
        return torch.nn.LSTM
    else:
        return torch.nn.GRU


WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "dwconv2d4": DepthwiseConv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
}

WENET_EMB_CLASSES = {
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
