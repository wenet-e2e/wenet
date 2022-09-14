import torch
import torch.nn as nn
from typing import Tuple
from wenet.squeezeformer.utils import ResidualModule
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4, TimeReductionLayer
from wenet.squeezeformer.encoder_layer import SqueezeformerEncoderLayer
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.convolution import ConvolutionModule
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask
from wenet.transformer.activations import Swish
from wenet.utils.common import get_activation


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
            input_dropout_rate: float = 0.1,
            pos_enc_layer_type: str = "rel_pos",
            do_rel_shift: bool = True,
            feed_forward_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            cnn_module_kernel: int = 31,
            cnn_norm_type: str = "batch_norm",
            dropout: float = 0.1,
            causal: bool = False,
            adaptive_scale: bool = True,
            activation_type: str = "swish",
            init_weights: bool = True,
            global_cmvn: torch.nn.Module = None,
            normalize_before: bool = False,
            use_dynamic_chunk: bool = False,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_left_chunk: bool = False
    ):
        """Construct SqueezeformerEncoder

                Args:
                    input_size to use_dynamic_chunk, see in Transformer BaseEncoder.
                    encoder_dim (int): The hidden dimension of encoder layer.
                    output_size (int): The output dimension of final projection layer.
                    attention_heads (int): Num of attention head in attention module.
                    num_blocks (int): Num of encoder layers.
                    reduce_idx (int): reduce layer index, from 40ms to 80ms per frame.
                    recover_idx (int): recover layer index, from 80ms to 40ms per frame.
                    feed_forward_expansion_factor (int): Enlarge coefficient of FFN layer.
                    input_dropout_rate (float): Dropout rate of input projection layer.
                    pos_enc_layer_type (str): Self attention type.
                    do_rel_shift (bool): Whether to do relative shift operation on rel-attention module.
                    cnn_module_kernel (int): Kernel size of CNN module.
                    activation_type (str): Encoder activation function type.
                    use_cnn_module (bool): Whether to use convolution module.
                    cnn_module_kernel (int): Kernel size of convolution module.
                    adaptive_scale (bool): Whether to use adaptive scale.
                    init_weights (bool): Whether to initialize weights.
                    causal (bool): whether to use causal convolution or not.
                """
        super(SqueezeformerEncoder, self).__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx = reduce_idx
        self.recover_idx = recover_idx
        self._output_size = output_size
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
                do_rel_shift,
                adaptive_scale,
                init_weights
            )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            encoder_dim,
            encoder_dim * feed_forward_expansion_factor,
            feed_forward_dropout_rate,
            activation,
            adaptive_scale,
            init_weights
        )

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (encoder_dim, cnn_module_kernel, activation,
                                  cnn_norm_type, causal, adaptive_scale, init_weights)

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
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        convolution_layer(*convolution_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        normalize_before,
                        dropout,
                        concat_after
                    ))
            elif reduce_idx <= layer_id < recover_idx:
                self.encoders.append(
                    ResidualModule(SqueezeformerEncoderLayer(
                        encoder_dim,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        convolution_layer(*convolution_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        normalize_before,
                        dropout,
                        concat_after
                    )))
            else:
                self.encoders.append(
                    SqueezeformerEncoderLayer(
                        encoder_dim,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        convolution_layer(*convolution_layer_args),
                        positionwise_layer(*positionwise_layer_args),
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
