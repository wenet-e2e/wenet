""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
from wenet.cif.predictor import Predictor
from wenet.paraformer.ali_paraformer.attention import (DummyMultiHeadSANM,
                                                       MultiHeadAttentionCross,
                                                       MultiHeadedAttentionSANM
                                                       )
from wenet.paraformer.search import paraformer_beam_search, paraformer_greedy_search
from wenet.transformer.search import DecodeResult
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import make_non_pad_mask


class LFR(torch.nn.Module):

    def __init__(self, m: int = 7, n: int = 6) -> None:
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.

        """
        super().__init__()

        self.m = m
        self.n = n

        self.left_padding_nums = math.ceil((self.m - 1) // 2)

    def forward(self, input: torch.Tensor,
                input_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, D = input.size()
        n_lfr = torch.ceil(input_lens / self.n).to(input_lens.dtype)
        # print(n_lfr)
        # right_padding_nums >= 0
        prepad_nums = input_lens + self.left_padding_nums

        right_padding_nums = torch.where(
            self.m >= (prepad_nums - self.n * (n_lfr - 1)),
            self.m - (prepad_nums - self.n * (n_lfr - 1)),
            0,
        )
        T_all = self.left_padding_nums + input_lens + right_padding_nums

        new_len = T_all // self.n

        T_all_max = T_all.max().int()

        tail_frames_index = (input_lens - 1).view(B, 1, 1).repeat(1, 1,
                                                                  D)  # [B,1,D]

        tail_frames = torch.gather(input, 1, tail_frames_index)
        tail_frames = tail_frames.repeat(1, right_padding_nums.max().int(), 1)
        head_frames = input[:, 0:1, :].repeat(1, self.left_padding_nums, 1)

        # stack
        input = torch.cat([head_frames, input, tail_frames], dim=1)

        index = torch.arange(T_all_max,
                             device=input.device,
                             dtype=input_lens.dtype).unsqueeze(0).repeat(
                                 B, 1)  # [B, T_all_max]
        # [B, T_all_max]
        index_mask = index < (self.left_padding_nums + input_lens).unsqueeze(1)

        tail_index_mask = torch.logical_not(
            index >= (T_all.unsqueeze(1))) & index_mask
        tail = torch.ones(T_all_max,
                          dtype=input_lens.dtype,
                          device=input.device).unsqueeze(0).repeat(B, 1) * (
                              T_all_max - 1)  # [B, T_all_max]
        indices = torch.where(torch.logical_or(index_mask, tail_index_mask),
                              index, tail)
        input = torch.gather(input, 1, indices.unsqueeze(2).repeat(1, 1, D))

        input = input.unfold(1, self.m, step=self.n).transpose(2, 3)
        # new len
        return input.reshape(B, -1, D * self.m), new_len


class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 idim,
                 hidden_units,
                 dropout_rate,
                 adim=None,
                 activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units,
                                   idim if adim is None else adim,
                                   bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = torch.nn.LayerNorm(hidden_units)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))


class SinusoidalPositionEncoder(torch.nn.Module):
    """https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/embedding.py#L387
    """

    def __int__(self):
        pass

    def encode(self,
               positions: torch.Tensor,
               depth: int,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype,
                         device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) *
            (-log_timescale_increment))
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time),
                              torch.cos(scaled_time)],
                             dim=2)
        return encoding.to(dtype)

    def forward(self, x):
        _, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim,
                                        x.dtype).to(x.device)

        return x + position_encoding


class AliParaformerEncoderLayer(TransformerEncoderLayer):

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 in_size: int = 256):
        """ Resize input in_size to size
        """
        super().__init__(size, self_attn, feed_forward, dropout_rate,
                         normalize_before)
        self.in_size = in_size
        self.size = size
        del self.norm1
        self.norm1 = torch.nn.LayerNorm(in_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, cache=att_cache)
        if self.in_size == self.size:
            x = residual + self.dropout(x_att)
        else:
            x = self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


class SanmEncoder(BaseEncoder):

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0,
                 input_layer: str = "conv2d",
                 pos_enc_layer_type: str = "abs_pos",
                 normalize_before: bool = True,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False,
                 kernel_size: int = 11,
                 sanm_shfit: int = 0):
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)
        del self.embed
        self.embed = SinusoidalPositionEncoder()

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        self.encoders0 = torch.nn.ModuleList([
            AliParaformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                in_size=input_size,
            )
        ])
        self.encoders = torch.nn.ModuleList([
            AliParaformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                PositionwiseFeedForward(
                    output_size,
                    linear_units,
                    dropout_rate,
                ),
                dropout_rate,
                normalize_before,
                in_size=output_size) for _ in range(num_blocks - 1)
        ])
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(output_size)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        masks_pad = make_non_pad_mask(xs_lens).unsqueeze(1)  # [B,1,T]
        xs = xs * self.output_size()**0.5
        xs = self.embed(xs)
        for layer in self.encoders0:
            xs, _, _, _ = layer(xs, masks_pad, None)
        for layer in self.encoders:
            xs, _, _, _ = layer(xs, masks_pad, None)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks_pad


class _Decoders3(torch.nn.Module):
    """Paraformer has a decoder3"""

    def __init__(self, hidden: int, pos_clss: torch.nn.Module) -> None:
        super().__init__()
        self.feed_forward = pos_clss
        self.norm1 = torch.nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm1(x))


class SANMDecoderLayer(DecoderLayer):

    def __init__(self,
                 size: int,
                 self_attn: Optional[torch.nn.Module],
                 src_attn: Optional[torch.nn.Module],
                 feed_forward: torch.nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True):
        super().__init__(size, self_attn, src_attn, feed_forward, dropout_rate,
                         normalize_before)
        # NOTE(Mddct): ali-Paraformer need eps=1e-12
        self.norm1 = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm2 = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm3 = torch.nn.LayerNorm(size, eps=1e-12)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        x = tgt
        if self.self_attn is not None:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            tgt_q = tgt
            x = self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0]
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask)[0])

        return x, tgt_mask, memory, memory_mask


class SanmDecoer(TransformerDecoder):

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0,
        src_attention_dropout_rate: float = 0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,
        att_layer_num: int = 16,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
    ):
        super().__init__(vocab_size, encoder_output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, self_attention_dropout_rate,
                         src_attention_dropout_rate, input_layer,
                         use_output_layer, normalize_before, src_attention)
        del self.embed
        del self.decoders
        self.decoders = torch.nn.ModuleList([
            SANMDecoderLayer(
                encoder_output_size,
                DummyMultiHeadSANM(attention_heads, encoder_output_size,
                                   encoder_output_size, dropout_rate,
                                   kernel_size, sanm_shfit),
                MultiHeadAttentionCross(attention_heads, encoder_output_size,
                                        encoder_output_size, dropout_rate,
                                        kernel_size, sanm_shfit,
                                        encoder_output_size),
                PositionwiseFeedForwardDecoderSANM(encoder_output_size,
                                                   linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
            ) for _ in range(att_layer_num)
        ])
        # NOTE(Mddct): att_layer_num == num_blocks in released pararformer model
        assert att_layer_num == num_blocks

        # NOTE(Mddct): Paraformer has a deocder3
        self.decoders3 = torch.nn.ModuleList([
            _Decoders3(
                encoder_output_size,
                PositionwiseFeedForwardDecoderSANM(encoder_output_size,
                                                   linear_units, dropout_rate))
        ])

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_mask: torch.Tensor,
        sematic_embeds: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ only for inference now
        """
        ys_pad_mask = make_non_pad_mask(ys_pad_lens).unsqueeze(1)

        x = sematic_embeds
        for layer in self.decoders:
            x, _, _, _ = layer(x, ys_pad_mask, encoder_out, encoder_out_mask)

        for layer in self.decoders3:
            x = layer(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, torch.tensor(0.0), ys_pad_lens


class AliParaformer(torch.nn.Module):

    def __init__(self, encoder: SanmEncoder, decoder: SanmDecoer,
                 predictor: Predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.lfr = LFR()
        self.sos = 1
        self.eos = 2

    @torch.jit.ignore(drop=True)
    def forward(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError

    @torch.jit.export
    def forward_paraformer(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, features_lens = self.lfr(speech, speech_lengths)
        features_lens = features_lens.to(speech_lengths.dtype)
        # encoder
        encoder_out, encoder_out_mask = self.encoder(features, features_lens)

        # cif predictor
        acoustic_embed, token_num, _, _ = self.predictor(encoder_out,
                                                         mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lengths.dtype)

        # decoder
        decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
                                         acoustic_embed, token_num)
        # decoder_out = decoder_out.log_softmax(dim=-1)
        return decoder_out, token_num

    def decode(self,
               methods: List[str],
               speech: torch.Tensor,
               speech_lengths: torch.Tensor,
               beam_size: int,
               decoding_chunk_size: int = -1,
               num_decoding_left_chunks: int = -1,
               ctc_weight: float = 0,
               simulate_streaming: bool = False,
               reverse_weight: float = 0) -> Dict[str, List[DecodeResult]]:
        decoder_out, decoder_out_lens = self.forward_paraformer(
            speech, speech_lengths)

        results = {}
        if 'paraformer_greedy_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_greedy_result = paraformer_greedy_search(
                decoder_out, decoder_out_lens)
            results['paraformer_greedy_search'] = paraformer_greedy_result
        if 'paraformer_beam_search' in methods:
            assert decoder_out is not None
            assert decoder_out_lens is not None
            paraformer_beam_result = paraformer_beam_search(
                decoder_out,
                decoder_out_lens,
                beam_size=beam_size,
                eos=self.eos)
            results['paraformer_beam_search'] = paraformer_beam_result

        return results
