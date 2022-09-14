import torch
import torch.nn as nn
from typing import Tuple
from wenet.squeezeformer.utils import ResidualModule
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4, TimeReductionLayer
from wenet.squeezeformer.encoder_layer import SqueezeformerEncoderLayer
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.convolution import ConvolutionModule
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask
from wenet.transformer.activations import Swish


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
            do_rel_shift: bool = True,
            attention_dropout_rate: float = 0.1,
            cnn_module_kernel: int = 31,
            cnn_norm_type: str = "batch_norm",
            dropout: float = 0.1,
            causal: bool = False,
            adaptive_scale: bool = True,
            init_weights: bool = True,
            global_cmvn: torch.nn.Module = None,
            normalize_before: bool = False,
            use_dynamic_chunk: bool = False,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_left_chunk: bool = False
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx = reduce_idx
        self.recover_idx = recover_idx
        self._output_size = output_size
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        assert conv_expansion_factor == 2

        self.embed = DepthwiseConv2dSubsampling4(
            1, encoder_dim, RelPositionalEncoding(encoder_dim, dropout_rate=0.1)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * (((input_size - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_rate),
        )
        self.preln = nn.LayerNorm(encoder_dim)
        self.encoders = torch.nn.ModuleList()
        for layer_id in range(num_blocks):
            if layer_id < reduce_idx:
                self.encoders.append(
                    SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(
                            attention_heads, encoder_dim, attention_dropout_rate,
                            do_rel_shift=True, adaptive_scale=adaptive_scale, init_weights=init_weights),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        ConvolutionModule(
                            encoder_dim, cnn_module_kernel, Swish(), cnn_norm_type, causal=causal,
                            adaptive_scale=adaptive_scale, init_weights=True),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        normalize_before,
                        dropout,
                        concat_after
                    ))
            elif reduce_idx <= layer_id < recover_idx:
                self.encoders.append(
                    ResidualModule(SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(
                            attention_heads, encoder_dim, attention_dropout_rate,
                            do_rel_shift=True, init_weights=True, adaptive_scale=adaptive_scale),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        ConvolutionModule(
                            encoder_dim, cnn_module_kernel, Swish(), cnn_norm_type, causal=causal,
                            adaptive_scale=adaptive_scale, init_weights=True),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        normalize_before,
                        dropout,
                        concat_after
                    )))
            else:
                self.encoders.append(
                    SqueezeformerEncoderLayer(
                        encoder_dim,
                        RelPositionMultiHeadedAttention(
                            attention_heads, encoder_dim, attention_dropout_rate,
                            do_rel_shift=do_rel_shift, init_weights=True, adaptive_scale=adaptive_scale),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        ConvolutionModule(
                            encoder_dim, cnn_module_kernel, Swish(), cnn_norm_type, causal=causal,
                            adaptive_scale=adaptive_scale, init_weights=True),
                        PositionwiseFeedForward(
                            encoder_dim, hidden_units=encoder_dim * feed_forward_expansion_factor,
                            dropout_rate=feed_forward_dropout_rate, activation=Swish(),
                            adaptive_scale=adaptive_scale, init_weights=init_weights),
                        normalize_before,
                        dropout,
                        concat_after
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
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        xs_lens = chunk_masks.squeeze(1).sum(1)
        xs = self.input_proj(xs)
        xs = self.preln(xs)
        recover_tensor = torch.tensor(0.)
        recover_chunk_masks = torch.tensor(0.)
        recover_pos_emb = torch.tensor(0.)
        recover_mask_pad = torch.tensor(0.)
        for idx, layer in enumerate(self.encoders):
            if idx == self.reduce_idx:
                recover_tensor = xs
                recover_chunk_masks = chunk_masks
                recover_pos_emb = pos_emb
                recover_mask_pad = mask_pad
                xs, xs_lens = self.time_reduction_layer(xs, xs_lens)
                reduce_t = xs.size(1)
                pos_emb = pos_emb[:, :reduce_t, :]
                chunk_masks = chunk_masks[:, ::2, ::2]
                mask_pad = mask_pad[:, :, ::2]

            if idx == self.recover_idx:
                # recover output length for ctc decode
                xs = xs.unsqueeze(2)
                xs = xs.repeat(1, 1, 2, 1).flatten(1, 2)
                xs = self.time_recover_layer(xs)
                recover_t = recover_tensor.size(1)
                xs = recover_tensor + xs[:, :recover_t, :].contiguous()
                chunk_masks = recover_chunk_masks
                pos_emb = recover_pos_emb
                mask_pad = recover_mask_pad

            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, masks


if __name__ == '__main__':
    for T in range(64, 240):
        # T = 128 * 4
        # T = 65
        x = torch.rand(2, T, 80)
        length = torch.tensor([T, T // 2])
        model = SqueezeformerEncoder(
            use_dynamic_chunk=True, causal=True,
            use_dynamic_left_chunk=False, do_rel_shift=True)
        # model = SqueezeformerEncoder()
        # print(model)
        x = model(x, length, 0, -1)
        # print('x', x.size())
        print('x', x[0].size())
    # print('x', x[1].size())
    torch.jit.script(model)
