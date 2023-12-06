# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2023 NetEase Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Tuple

import torch

from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder


class DualTransformerEncoder(TransformerEncoder):
    """Transformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """ Construct DualTransformerEncoder
        Support both the full context mode and the streaming mode separately
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs,
                                              masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks,
                                              enable_full_context=False)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_full(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        for layer in self.encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


class DualConformerEncoder(ConformerEncoder):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
    ):
        """ Construct DualConformerEncoder
        Support both the full context mode and the streaming mode separately
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, positionwise_conv_kernel_size,
                         macaron_style, selfattention_layer_type,
                         activation_type, use_cnn_module, cnn_module_kernel,
                         causal, cnn_module_norm)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs,
                                              masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks,
                                              enable_full_context=False)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_full(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        for layer in self.encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
