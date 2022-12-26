# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2022 58.com(Wuba) Inc AI Lab.
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
# Modified from EfficientConformer(https://github.com/burchim/EfficientConformer)
#               Paper(https://arxiv.org/abs/2109.01163)

"""Encoder definition."""
from typing import Tuple, List, Optional

import torch
import logging
from typeguard import check_argument_types
import torch.nn.functional as F

from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.efficient_conformer.subsampling import Conv2dSubsampling2
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.efficient_conformer.attention import MultiHeadedAttention
from wenet.efficient_conformer.attention import RelPositionMultiHeadedAttention
from wenet.efficient_conformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import ConformerEncoderLayer

from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask

from wenet.efficient_conformer.attention import GroupedRelPositionMultiHeadedAttention
from wenet.efficient_conformer.encoder_layer import StrideConformerEncoderLayer


class EfficientConformerEncoder(torch.nn.Module):
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
        stride_layer_idx: list = (3),
        stride: list = (2),
        group_layer_idx: list = (0, 1, 2, 3),
        group_size: int = 3,
        stride_kernel: bool = True,
        efficient_conf = None
    ):
        """Construct Efficient Conformer Encoder

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

            stride_layer_idx (list): layer id with StrideConv
            stride (list): stride size of each StrideConv in efficient conformer
            group_layer_idx (list): layer id with GroupedAttention
            group_size (int): group size of every GroupedAttention layer
            stride_kernel (bool): default True. True: recompute cnn kernels with stride.
            efficient_conf (dict):
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
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        logging.info(f"input_layer = {input_layer}, subsampling_class = {subsampling_class}")

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )
        self.input_layer = input_layer
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        activation = get_activation(activation_type)
        self.num_blocks = num_blocks
        self.attention_heads = attention_heads
        self.cnn_module_kernel = cnn_module_kernel
        self.global_chunk_size = 0

        # efficient conformer configs
        assert len(stride) == len(stride_layer_idx)
        self.cnn_module_kernels = [cnn_module_kernel]  # kernel size of each StridedConv
        for i in stride:
            if stride_kernel:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1]//i)
            else:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1])
        logging.info(f"stride_layer_idx= {stride_layer_idx}, stride = {stride}, "
                     f"cnn_module_kernel = {self.cnn_module_kernels}")
        self.stride_layer_idx = stride_layer_idx   # layer id with StrideConv
        self.stride = stride                       # stride size of each StrideConv
        self.group_layer_idx = group_layer_idx     # layer id with GroupedAttention
        self.grouped_size = group_size             # group size of every GroupedAttention layer
        logging.info(f"group_layer_idx = {group_layer_idx}, "
                     f"grouped_size = {self.grouped_size}")

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

        # encoder definition
        index = 0
        layers = []
        for i in range(num_blocks):
            # self-attention module definition
            if i in self.group_layer_idx:
                encoder_selfattn_layer = GroupedRelPositionMultiHeadedAttention
                encoder_selfattn_layer_args = (
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    self.grouped_size)
            else:
                if pos_enc_layer_type == "no_pos":
                    encoder_selfattn_layer = MultiHeadedAttention
                else:
                    encoder_selfattn_layer = RelPositionMultiHeadedAttention
                encoder_selfattn_layer_args = (
                    attention_heads,
                    output_size,
                    attention_dropout_rate)

            # conformer module definition
            if i in self.stride_layer_idx:
                # conformer block with downsampling
                convolution_layer_args_stride = (output_size, self.cnn_module_kernels[index], activation,
                                         cnn_module_norm, causal, True, stride[index])
                logging.info(f"convolution_layer_args_stride = {convolution_layer_args_stride}")
                layers.append(StrideConformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(
                        *positionwise_layer_args) if macaron_style else None,
                    convolution_layer(
                        *convolution_layer_args_stride) if use_cnn_module else None,
                    torch.nn.AvgPool1d(kernel_size=stride[index], stride=stride[index], padding=0,
                                       ceil_mode=True, count_include_pad=False),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ))
                index = index + 1
            else:
                # conformer block
                convolution_layer_args_normal = (output_size, self.cnn_module_kernels[index], activation,
                                                 cnn_module_norm, causal)
                logging.info(f"convolution_layer_args_normal = {convolution_layer_args_normal}")
                layers.append(ConformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(
                        *positionwise_layer_args) if macaron_style else None,
                    convolution_layer(
                        *convolution_layer_args_normal) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ))

        self.encoders = torch.nn.ModuleList(layers)

    def set_global_chunk_size(self, chunk_size):
        """Used in ONNX export.
        """
        logging.info(f"set global chunk size: {chunk_size}, default is 0.")
        self.global_chunk_size = chunk_size

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
            chunk_masks = add_optional_chunk_mask(xs, masks,
                                                  self.use_dynamic_chunk,
                                                  self.use_dynamic_left_chunk,
                                                  decoding_chunk_size,
                                                  self.static_chunk_size,
                                                  num_decoding_left_chunks)
            index = 0 # traverse stride
            for i, layer in enumerate(self.encoders):
                # layer return : x, mask, new_att_cache, new_cnn_cache
                xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
                if i in self.stride_layer_idx:
                    masks = masks[:, :, ::self.stride[index]]
                    # chunk_masks = add_optional_chunk_mask(xs, masks,
                    #                                 self.use_dynamic_chunk,
                    #                                 self.use_dynamic_left_chunk,
                    #                                 decoding_chunk_size,
                    #                                 self.static_chunk_size,
                    #                                num_decoding_left_chunks)
                    chunk_masks = chunk_masks[:, ::self.stride[index], ::self.stride[index]]
                    mask_pad = masks
                    pos_emb = pos_emb[:, ::self.stride[index], :]
                    index = index + 1

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
        att_cache_shape: torch.Tensor =  torch.ones((0, 0), dtype=torch.int64),
        cnn_cache_shape: torch.Tensor =  torch.ones((0, 0), dtype=torch.int64),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input
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
            att_mask : mask matrix of self attention
            att_cache_shape :
            cnn_cache_shape :

        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache

        """
        assert xs.size(0) == 1

        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)  # (1, 1, xs-time)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers = att_cache.size(0)
        # cache_t1 = att_cache.size(2)
        cache_t1 = att_cache_shape[0][2].item() if att_cache_shape.size(0) > 0 \
                                                        and att_cache.size(2) > 0 else att_cache.size(2)

        # for ONNX export， padding xs to ChunkSize
        if self.global_chunk_size > 0:
            real_len = xs.size(1)
            xs = F.pad(xs, (0, 0, 0, self.global_chunk_size - real_len), value=0.0)
            tmp_zeros = torch.zeros(att_mask.shape, dtype=torch.bool)
            att_mask[:, :, required_cache_size+real_len+1:] = tmp_zeros[:, :, required_cache_size+real_len+1:]

        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        att_cache_padding_len = required_cache_size
        cnn_cache_padding_len = self.cnn_module_kernel - 1

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        r_att_cache_shape = torch.ones(self.num_blocks, 4, dtype=torch.int64) * 65535
        r_cnn_cache_shape = torch.ones(self.num_blocks, 4, dtype=torch.int64) * 65535
        mask_pad = att_mask[:,:,-chunk_size:]
        for i, layer in enumerate(self.encoders):
            cache_t1 = att_cache_shape[i][2].item() if att_cache_shape.size(0) > 0 \
                                                       and att_cache.size(2) > 0 else att_cache.size(2)
            pos_emb = self.embed.position_encoding(
                   offset=int(offset-cache_t1) if offset > cache_t1 else 0,
                   size=xs.size(1)+cache_t1)

            # use "att_cache_shape" and "cnn_cache_shape" record real shape of "att_cache" and "cnn_cache"
            if att_cache_shape.size(0) > 0:
                i_att_cache = att_cache[
                      i:i + 1, :att_cache_shape[i][1], :att_cache_shape[i][2],
                      :att_cache_shape[i][3]] if elayers > 0 else att_cache
                i_cnn_cache = cnn_cache[
                      i, :cnn_cache_shape[i][1], :cnn_cache_shape[i][2],
                      :cnn_cache_shape[i][3]] if cnn_cache.size(0) > 0 else cnn_cache
            else:
                i_att_cache = att_cache[i:i + 1,:,:,:] if elayers > 0 else att_cache
                i_cnn_cache = cnn_cache[i,:,:,:] if cnn_cache.size(0) > 0 else cnn_cache

            # old new_att_cache: [ batch, head, time2, outdim//head * 2 ]
            # new new_att_cache: [ batch, time2, outdim*2 ]
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, pos_emb,
                mask_pad=mask_pad,
                att_cache=i_att_cache,
                cnn_cache=i_cnn_cache)

            if i in self.stride_layer_idx:
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]
                mask_pad = mask_pad[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]

            new_att_cache = new_att_cache[:, :, next_cache_start:, :] # [batch, head, time2, outdim]
            new_cnn_cache = new_cnn_cache.unsqueeze(0)                # [1, batch, outdim, cache_t2]

            # update real shape of att_cache and cnn_cache
            for ishape in range(len(new_att_cache.shape)):
                r_att_cache_shape[i][ishape] = new_att_cache.shape[ishape]
                r_cnn_cache_shape[i][ishape] = new_cnn_cache.shape[ishape]
            r_att_cache.append(new_att_cache)
            r_cnn_cache.append(new_cnn_cache)

        if self.normalize_before:
            xs = self.after_norm(xs)

        for i in range(len(r_att_cache)):
            r_att_cache[i] = F.pad(r_att_cache[i], (0, 0, 0, att_cache_padding_len-r_att_cache[i].shape[2]))
            r_cnn_cache[i] = F.pad(r_cnn_cache[i], (0, cnn_cache_padding_len-r_cnn_cache[i].shape[3]))

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return xs, r_att_cache, r_cnn_cache, r_att_cache_shape, r_cnn_cache_shape

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
        use_onnx = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
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
            decoding_chunk_size (int): decoding chunk size
            num_decoding_left_chunks (int):
            use_onnx (bool): True for simulating ONNX model inference.
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        if use_onnx:
            logging.info(f"Simulating for ONNX runtime ...")
            att_cache: torch.Tensor = torch.zeros(
               (self.num_blocks, self.attention_heads, required_cache_size,
                self.output_size()//self.attention_heads*2),
               device=xs.device)
            cnn_cache: torch.Tensor = torch.zeros(
               (self.num_blocks, 1, self.output_size(), self.cnn_module_kernel-1),
               device=xs.device)
            att_cache_shape = torch.ones(self.num_blocks, 4, dtype=torch.int64) * 65535
            cnn_cache_shape = torch.ones(self.num_blocks, 4, dtype=torch.int64) * 65535
            self.set_global_chunk_size(chunk_size=18)
        else:
            logging.info(f"Simulating for JIT runtime ...")
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
            att_cache_shape: torch.Tensor =  torch.ones((0, 0), dtype=torch.int64)
            cnn_cache_shape: torch.Tensor =  torch.ones((0, 0), dtype=torch.int64)

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            logging.info(f"-->> frame chunk msg: cur={cur}, end={end}, num_frames={end-cur}, "
                         f"decoding_window={decoding_window}")
            if use_onnx:
                att_mask: torch.Tensor = torch.ones(
                    (1, 1, required_cache_size + decoding_chunk_size),
                    dtype=torch.bool, device=xs.device)
                if cur == 0:
                    att_mask[:, :, :required_cache_size] = 0
            else:
                att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool, device=xs.device)

            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache, att_cache_shape, cnn_cache_shape) = self.forward_chunk(
                chunk_xs, offset, required_cache_size,
                att_cache, cnn_cache,
                att_mask, att_cache_shape, cnn_cache_shape
            )
            outputs.append(y)
            if self.input_layer == "linear":
                t_hat = chunk_xs.size(1)
            elif self.input_layer == "conv2d2":
                t_hat = (chunk_xs.size(1) - 1) // 2
            elif self.input_layer == "conv2d":
                t_hat = ((chunk_xs.size(1) - 1) // 2 - 1) // 2
            elif self.input_layer == "conv2d6":
                t_hat = ((chunk_xs.size(1) - 1) // 2 - 2) // 3
            elif self.input_layer == "conv2d8":
                t_hat = (((chunk_xs.size(1) - 1) // 2 - 1) // 2 - 1) // 2
            else:
                t_hat = y.size(1)
            offset += t_hat

        ys = torch.cat(outputs, 1)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        return ys, masks
