# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
from typing import List

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.encoder_layer import SqueezeformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.transformer.subsampling import TimeReduction2
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask


class BaseEncoder(torch.nn.Module):
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
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        time_reduce_idx: List[int] = None,
        time_recover_idx: List[int] = None,
        input_layer_depthwise: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            time_reduce_idx: layer indices list where 2x-time-reduction is applied
            time_recover_idx: layer indices list where 2x-time-recover is applied
            input_layer_depthwise (bool): to make second conv layer
                depthwise-separable or not.
                (This reduces the number of floating point operations.)
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
            input_layer_depthwise,
        )

        # Codes for 2x time-reduction and time-recovery
        time_reduce_idx = (
            [int(idx) for idx in time_reduce_idx] if time_reduce_idx is not None else []
        )
        # NOTE(hslee):
        #   adding a fake idx -1 is just to avoid error in torch.jit.script(),
        #   since an list attribute with empty elements found to make an error.
        #   If there's any better detour, please fix this part.
        time_reduce_idx = [-1] + time_reduce_idx
        self.time_reduce_idx = time_reduce_idx
        time_recover_idx = (
            [int(idx) for idx in time_recover_idx]
            if time_recover_idx is not None
            else []
        )
        time_recover_idx = [-1] + time_recover_idx
        self.time_recover_idx = time_recover_idx
        self.time_reduce_layers = []
        self.time_recover_layers = []
        if time_reduce_idx is not None and len(time_reduce_idx) > 0:
            assert len(time_recover_idx) > 0
            assert len(time_reduce_idx) == len(time_recover_idx)
            for idx in time_reduce_idx:
                if idx == -1:
                    continue
                self.time_reduce_layers.append(
                    TimeReduction2(
                        output_size,
                        output_size,
                        0.0,
                    )
                )
            for idx in time_recover_idx:
                if idx == -1:
                    continue
                self.time_recover_layers.append(
                    torch.nn.Linear(
                        output_size,
                        output_size,
                    )
                )
            self.time_reduce_layers = torch.nn.ModuleList(self.time_reduce_layers)
            self.time_recover_layers = torch.nn.ModuleList(self.time_recover_layers)

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

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
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )

        time_reduce_residuals = torch.jit.annotate(
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], []
        )
        time_reduce_level: int = 0
        for i, layer in enumerate(self.encoders):
            if len(self.time_reduce_idx) > 0 and i in self.time_reduce_idx:
                time_reduce_residuals.append((xs, mask_pad, chunk_masks, pos_emb))
                for i, reduce_layer in enumerate(self.time_reduce_layers):
                    if time_reduce_level == i:
                        xs, mask_pad, chunk_masks, pos_emb = reduce_layer(
                            xs, mask_pad, chunk_masks, pos_emb
                        )
                time_reduce_level += 1
            if len(self.time_reduce_idx) > 0 and i in self.time_recover_idx:
                time_reduce_level -= 1
                residual_xs, mask_pad, chunk_masks, pos_emb = time_reduce_residuals[
                    time_reduce_level
                ]
                B, T_, D = xs.size()
                xs = xs.unsqueeze(2).tile((1, 1, 2, 1))  # (B,T',2,D')
                xs = xs.reshape(B, -1, D)[
                    :, : residual_xs.size(1), :
                ]  # (B,T,D'), T=floor(T*2)
                for i, recover_layer in enumerate(self.time_recover_layers):
                    if time_reduce_level == i:
                        xs = recover_layer(xs)
                xs += residual_xs  # (B,T,D)
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size
        )
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        # NOTE(hslee):
        #  Current version of att_cache does not consider time_reduction for caching.
        #  It leads to mismatch between forward (train) and forward_chunk (stream),
        #   since the length of past attention key becomes
        #   longer than the length used in forward().
        #  To resolve the mismatch, required_cache_size should be
        #   reduced as time_reduce_level increases.
        #  However, I just left required_cache_size as it was for the following reasons:
        #   - giving longer past context for attention keys may improve WER/CER.
        #   - (when dynamic_chunk==true) time reduced layers are
        #      already trained for various length of past context.
        #   - implementation of variable time reduction rate
        #      for att_cache seems a bit tricky.
        mask_pad_fake = torch.zeros((0, 0, 0), dtype=xs.dtype, device=xs.device)
        time_reduce_residuals = torch.jit.annotate(
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], []
        )
        time_reduce_level: int = 0
        for i, layer in enumerate(self.encoders):
            if len(self.time_reduce_idx) > 0 and i in self.time_reduce_idx:
                time_reduce_residuals.append((xs, att_mask, pos_emb))
                for i, reduce_layer in enumerate(self.time_reduce_layers):
                    if time_reduce_level == i:
                        xs, _, att_mask, pos_emb = reduce_layer(
                            xs, mask_pad_fake, att_mask, pos_emb
                        )
                time_reduce_level += 1
            if len(self.time_reduce_idx) > 0 and i in self.time_recover_idx:
                time_reduce_level -= 1
                residual_xs, chunk_masks, pos_emb = time_reduce_residuals[
                    time_reduce_level
                ]
                B, T_, D = xs.size()
                xs = xs.unsqueeze(2).tile((1, 1, 2, 1))  # (B,T',2,D')
                xs = xs.reshape(B, -1, D)[
                    :, : residual_xs.size(1), :
                ]  # (B,T,D'), T=floor(T*2)
                for i, recover_layer in enumerate(self.time_recover_layers):
                    if time_reduce_level == i:
                        xs = recover_layer(xs)
                xs = xs + residual_xs  # (B,T,D)
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache[i : i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache
            )
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks


class TransformerEncoder(BaseEncoder):
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
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        assert check_argument_types()
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            concat_after,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
        )
        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    output_size,
                    MultiHeadedAttention(
                        attention_heads, output_size, attention_dropout_rate
                    ),
                    PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )


class ConformerEncoder(BaseEncoder):
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
        concat_after: bool = False,
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
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            concat_after,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
        )
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )

        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args)
                    if macaron_style
                    else None,
                    convolution_layer(*convolution_layer_args)
                    if use_cnn_module
                    else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )


class SqueezeformerEncoder(BaseEncoder):
    """Squeezeformer encoder module."""

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
        normalize_before: bool = False,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        time_reduce_idx: List[int] = None,
        time_recover_idx: List[int] = None,
        input_layer_depthwise: bool = True,
    ):
        """Construct SqueezeformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            concat_after,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
            time_reduce_idx,
            time_recover_idx,
            input_layer_depthwise,
        )
        activation = get_activation(activation_type)

        assert (
            not normalize_before
        ), "Squeezeformer must use PostLN rather than PreLN"
        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
            True,
            False,  # use_glu=False
        )

        self.encoders = torch.nn.ModuleList(
            [
                SqueezeformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    convolution_layer(*convolution_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
