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
from typing import Tuple, Optional

import torch
import math


from wenet.chunkformer.attention import ChunkAttentionWithRelativeRightContext
from wenet.chunkformer.convolution import ChunkConvolutionModule
from wenet.chunkformer.embedding import RelPositionalEncodingWithRightContext
from wenet.chunkformer.encoder_layer import ChunkFormerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.chunkformer.subsampling import DepthwiseConvSubsampling
from wenet.utils.mask import make_pad_mask
from wenet.transformer.encoder import WENET_ACTIVATION_CLASSES, BaseEncoder
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_EMB_CLASSES, WENET_MLP_CLASSES,
                                     WENET_NORM_CLASSES,
                                     WENET_SUBSAMPLE_CLASSES)
class ChunkFormerEncoder(BaseEncoder):
    """ChunkFormer encoder module."""
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
        input_layer: str = "dw_striding",
        pos_enc_layer_type: str = "chunk_rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        selfattention_layer_type: str = "chunk_rel_seflattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        dynamic_conv: bool = False,
        layer_norm_type: str = 'layer_norm',
        gradient_checkpointing: bool = False,
        final_norm: bool = True,
        norm_eps: float = 1e-5,
        use_sdpa: bool = False,


    ):
        """Construct ChunkFormerEncoder

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
        torch.nn.Module.__init__(self)        
        assert  selfattention_layer_type == "chunk_rel_seflattn", f"ChunkFormer requires chunk_rel_seflattn, but {pos_enc_layer_type} is given"
        assert  pos_enc_layer_type == "chunk_rel_pos", f"ChunkFormer requires chunk_rel_pos, but {pos_enc_layer_type} is given"
        assert input_layer == "dw_striding", f"ChunkFormer requires input_layer, but {input_layer} is given"


        self._output_size = output_size

        self.global_cmvn = global_cmvn

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.final_norm = final_norm
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size,
                                                              eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa
        
        self._output_size = output_size
        self.global_cmvn = global_cmvn
        # NOTE(Mddct): head_dim == output_size // attention_heads for most of
        #    speech tasks,  but for other task (LLM),
        #    head_dim == hidden_size * attention_heads. refactor later

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.final_norm = final_norm
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size,
                                                              eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing



        self.cnn_module_kernel = cnn_module_kernel
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        self.num_blocks = num_blocks
        self.dynamic_conv = dynamic_conv
        self.input_size = input_size
        self.attention_heads = attention_heads
            
        self.embed = DepthwiseConvSubsampling(
            subsampling=input_layer,
            subsampling_rate=8,
            feat_in=input_size,
            feat_out=output_size,
            conv_channels=output_size,
            pos_enc_class=RelPositionalEncodingWithRightContext(output_size, positional_dropout_rate),
            subsampling_conv_chunking_factor=1,
            activation=torch.nn.ReLU(),
        )


        encoder_selfattn_layer = ChunkAttentionWithRelativeRightContext
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
        convolution_layer = ChunkConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, True, dynamic_conv)

        self.encoders = torch.nn.ModuleList([
            ChunkFormerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before
            ) for _ in range(num_blocks)
        ])

    def forward_parallel_chunk(
        self,
        xs,
        xs_origin_lens,
        chunk_size: int = -1,
        left_context_size: int = -1,
        right_context_size: int = -1,
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        truncated_context_size:int = 0,
        offset: torch.Tensor = torch.zeros(0),
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
        assert offset.shape[0] == len(xs), f"{offset.shape[0]} - {len(xs)}"
        
        # --------------------------Chunk Batching-------------------------------------------
        subsampling = self.embed.subsampling_factor
        context = self.embed.right_context + 1 # Add current frame
        size = (chunk_size - 1) * subsampling + context
        step = subsampling * chunk_size
        device = xs_origin_lens.device

        conv_lorder = self.cnn_module_kernel // 2

        upper_bounds = []
        lower_bounds = []
        upper_bounds_conv = []
        lower_bounds_conv = []
        x_pad = []
        xs_lens = []
        n_chunks = []
        for xs_origin_len, x, offs in zip(xs_origin_lens, xs, offset): # cost O(input_batch_size | ccu)
            x = x.to(device)
            if x.size(0) >= size:
                n_frames_pad = (step - ((x.size(0) - size) %  step)) % step
            else:
                n_frames_pad = size - x.size(0)
            x = torch.nn.functional.pad(x, (0, 0, 0, n_frames_pad)) # (T, 80)
            n_chunk = ((x.size(0) - size) // step) + 1
            x = x.unfold(0, size=size, step=step) # [n_chunk, 80, size]
            x = x.transpose(2, 1)

            max_len = 1  + (xs_origin_len - context)//subsampling
            upper_bound = chunk_size + right_context_size + torch.arange(0, 1 + (xs_origin_len + n_frames_pad - context)//subsampling, 1 + (size - context)//subsampling, device=device)
            lower_bound = upper_bound - max_len
            upper_bound += offs
            
            upper_bound_conv = chunk_size + conv_lorder + torch.arange(0, 1  + (xs_origin_len + n_frames_pad - context)//subsampling, 1 + (size - context)//subsampling, device=device)
            lower_bound_conv = torch.maximum(upper_bound_conv - max_len, torch.full_like(upper_bound_conv, conv_lorder - right_context_size))
            upper_bound_conv += offs


            xs_lens += [size] * (n_chunk - 1) + [size - n_frames_pad]
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
            upper_bounds_conv.append(upper_bound_conv)
            lower_bounds_conv.append(lower_bound_conv)
            x_pad.append(x)
            n_chunks.append(n_chunk)


        xs = torch.cat(x_pad, dim=0).to(device)
        xs_lens = torch.tensor(xs_lens).to(device)
        upper_bounds = torch.cat(upper_bounds).unsqueeze(1).to(device)
        lower_bounds = torch.cat(lower_bounds).unsqueeze(1).to(device)
        upper_bounds_conv = torch.cat(upper_bounds_conv).unsqueeze(1).to(device)
        lower_bounds_conv = torch.cat(lower_bounds_conv).unsqueeze(1).to(device)


        # forward model
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)


        xs, pos_emb, xs_lens = self.embed(xs, xs_lens, offset=left_context_size, right_context_size=right_context_size)
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)  # (B, 1, T)


        mask_pad = torch.arange(0, conv_lorder + chunk_size + conv_lorder, device=masks.device).unsqueeze(0).repeat(xs.size(0), 1) # [B, left_context_size + chunksize]
        mask_pad = (lower_bounds_conv <= mask_pad) & (mask_pad < upper_bounds_conv)
        mask_pad = mask_pad.flip(-1).unsqueeze(1)
        att_mask = torch.arange(0, left_context_size + chunk_size + right_context_size, device=masks.device).unsqueeze(0).repeat(xs.size(0), 1) # [B, left_context_size + chunksize]
        att_mask = (lower_bounds <= att_mask) & (att_mask < upper_bounds)
        att_mask = att_mask.flip(-1).unsqueeze(1)


        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer.forward_parallel_chunk(xs, att_mask, pos_emb, 
                mask_pad=mask_pad,
                right_context_size=right_context_size,
                left_context_size=left_context_size,
                att_cache=att_cache[i].to(device) if att_cache.size(0) > 0 else att_cache,
                cnn_cache=cnn_cache[i].to(device) if cnn_cache.size(0) > 0 else cnn_cache,
                truncated_context_size=truncated_context_size

            )
            r_att_cache.append(new_att_cache)
            r_cnn_cache.append(new_cnn_cache)

        del att_cache
        del cnn_cache
        if self.normalize_before:
            xs = self.after_norm(xs)

        xs_lens = self.embed.calc_length(xs_origin_lens)
        offset += xs_lens


        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.stack(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.stack(r_cnn_cache, dim=0)
        return xs, xs_lens, n_chunks, r_att_cache, r_cnn_cache, offset
    