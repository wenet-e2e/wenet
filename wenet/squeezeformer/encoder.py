import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Dict
from wenet.squeezeformer.utils import ResidualModule, recover_resolution
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4, TimeReductionLayer
from wenet.squeezeformer.embedding import RelPositionalEncoding2 as RelPositionalEncoding
# from wenet.transformer.embedding import RelPositionalEncoding
from wenet.squeezeformer.encoder_layer import SqueezeformerEncoderLayer
from wenet.squeezeformer.attention import RelPositionMultiHeadedAttention2 as RelPositionMultiHeadedAttention
from wenet.squeezeformer.ffn import PositionwiseFeedForward
from wenet.squeezeformer.convolution import ConvolutionModule
from wenet.utils.mask import make_pad_mask


class SqueezeformerEncoder(nn.Module):
    def __init__(
            self,
            input_size: int = 80,
            encoder_dim: int = 256,
            output_size: int = 256,
            attention_heads: int = 4,
            num_blocks: int = 12,
            reduce_idx: int = 5,
            recover_idx: int = 11,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_rate: float = 0.1,
            feed_forward_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            cnn_module_kernel: int = 31,
            cnn_dropout_rate: float = 0.1,
            global_cmvn: torch.nn.Module = None,
            normalize_before: bool = False
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx = reduce_idx
        self.recover_idx = recover_idx
        self._output_size = output_size
        assert conv_expansion_factor == 2

        self.embed = DepthwiseConv2dSubsampling4(
            1, encoder_dim, RelPositionalEncoding(encoder_dim, dropout_rate=0.1)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * (((input_size - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_rate),
        )
        self.encoders = torch.nn.ModuleList()
        for layer_id in range(num_blocks):
            if layer_id < reduce_idx:
                self.encoders.append(
                    SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(attention_heads, encoder_dim, attention_dropout_rate),
                        PositionwiseFeedForward(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate),
                        ConvolutionModule(encoder_dim, cnn_module_kernel, conv_expansion_factor, cnn_dropout_rate),
                        normalize_before
                    ))
            elif reduce_idx <= layer_id < recover_idx:
                self.encoders.append(
                    ResidualModule(SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(attention_heads, encoder_dim, attention_dropout_rate),
                        PositionwiseFeedForward(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate),
                        ConvolutionModule(encoder_dim, cnn_module_kernel, conv_expansion_factor, cnn_dropout_rate),
                        normalize_before
                    )))
            else:
                self.encoders.append(
                    SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(attention_heads, encoder_dim, attention_dropout_rate),
                        PositionwiseFeedForward(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate),
                        ConvolutionModule(encoder_dim, cnn_module_kernel, conv_expansion_factor, cnn_dropout_rate),
                        normalize_before
                    ))
        self.time_reduction_layer = TimeReductionLayer(encoder_dim=encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.final_proj = None
        if output_size != encoder_dim:
            self.final_proj = nn.Linear(encoder_dim, output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
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
        # print('xs', xs.size())
        # print('masks', masks.size())
        # print('pos_emb', pos_emb.size())
        # mask_pad = masks  # (B, 1, T/subsample_rate)
        masks = torch.ones_like(masks)
        xs = self.input_proj(xs)
        recover_tensor = torch.tensor(0.)
        recover_mask = torch.tensor(0.)
        recover_pos_emb = torch.tensor(0.)
        for idx, layer in enumerate(self.encoders):
            if idx == self.reduce_idx:
                recover_tensor = xs
                recover_mask = masks
                recover_pos_emb = pos_emb
                xs, xs_lens = self.time_reduction_layer(xs, xs_lens)
                reduce_t = xs.size(1)
                # masks = masks[:, :, :reduce_t]
                pos_emb = pos_emb[:, :reduce_t, :]
                masks = masks[:, :, :reduce_t * 2]
                masks = masks[:, :, ::2]
                # pos_emb = pos_emb[:, :reduce_t * 4 - 2, :]
                # pos_emb = pos_emb[:, ::2, :]

            if idx == self.recover_idx:
                xs = recover_resolution(xs)
                recover_t = xs.size(1)
                xs = self.time_recover_layer(xs)
                xs += recover_tensor[:, :recover_t, :]
                xs_lens *= 2
                recoverd_t = xs.size(1)
                masks = recover_mask[:, :, :recoverd_t]
                pos_emb = recover_pos_emb[:, :recoverd_t, :]
                # masks = recover_mask[:, :, :recoverd_t]
                # pos_emb = recover_pos_emb[:, :recoverd_t * 2 - 1, :]

            # print('[I] idx: {}, xs: {}, masks: {}, pos: {}'.format(idx, xs.size(), masks.size(), pos_emb.size()))
            xs = layer(xs, masks, pos_emb)
            # print('[O] idx: {}, xs: {}'.format(idx, xs.size()))
        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, masks


if __name__ == '__main__':
    # for T in range(64, 240):
    T = 128 * 4
    x = torch.rand(2, T, 80)
    length = torch.tensor([T, T])
    model = SqueezeformerEncoder()
    print(model)
    x = model(x, length, 0, -1)
    # print('x', x.size())
    print('x', x[0].size())
    # print('x', x[1].size())
    # torch.jit.script(model)
