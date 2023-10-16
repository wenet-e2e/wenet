""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import argparse
from typing import Optional, Tuple
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn import TransformerDecoder
import yaml
from wenet.cif.predictor import Predictor, cif
from wenet.paraformer.experiment.attention import DummyMultiHeadSANM, MultiHeadAttentionCross, MultiHeadedAttentionSANM
from wenet.paraformer.experiment.lfr import LFR
from wenet.paraformer.experiment.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.cmvn import load_cmvn
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.mask import make_non_pad_mask


class SinusoidalPositionEncoder(torch.nn.Module):
    """https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/embedding.py#L387
    """

    def __int__(self):
        pass

    def encode(self,
               positions: torch.Tensor = None,
               depth: int = None,
               dtype: torch.dtype = torch.float32):
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
        return encoding.type(dtype)

    def forward(self, x):
        _, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim,
                                        x.dtype).to(x.device)

        return x + position_encoding


class EncoderLayerSANM(TransformerEncoderLayer):

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 in_size: int = 256):
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
            EncoderLayerSANM(
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
            EncoderLayerSANM(
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
        # masks = masks_pad * masks_pad.transpose(1, 2)  #[B,T,T]
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


class DecoderLayerSANM(DecoderLayer):

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
        if self.self_attn:
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
            DecoderLayerSANM(
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
    ) -> torch.Tensor:
        """ only for inference now
        """
        ys_pad_mask = make_non_pad_mask(ys_pad_lens).unsqueeze(1)

        x = sematic_embeds
        for layer in self.decoders:
            x, _, _, _ = layer(x, ys_pad_mask, encoder_out, encoder_out_mask)

        for layer in self.decoders3:
            # x, _, _, _ = layer(x, ys_pad_mask, encoder_out, encoder_out_mask)
            x = layer(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x


class PredictorV3(Predictor):

    def __init__(self,
                 idim,
                 l_order,
                 r_order,
                 threshold=1,
                 dropout=0.1,
                 smooth_factor=1,
                 noise_threshold=0,
                 tail_threshold=0.45,
                 cnn_groups: int = 1):
        super().__init__(idim, l_order, r_order, threshold, dropout,
                         smooth_factor, noise_threshold, tail_threshold)

        self.cif_conv1d = torch.nn.Conv1d(idim,
                                          idim,
                                          l_order + r_order + 1,
                                          groups=cnn_groups)

    def forward(self,
                hidden,
                target_label: Optional[torch.Tensor] = None,
                mask: torch.Tensor = ...,
                ignore_id: int = -1,
                mask_chunk_predictor: Optional[torch.Tensor] = None,
                target_label_length: Optional[torch.Tensor] = None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor -
                                          self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(
                1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden,
                                                             alphas,
                                                             token_num,
                                                             mask=mask)
        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak


class Paraformer(torch.nn.Module):

    def __init__(self, encoder: SanmEncoder, decoder: SanmDecoer,
                 predictor: Predictor) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

        self.lfr = LFR()

    def forward(
            self, speech: torch.Tensor,
            speech_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features, features_lens = self.lfr(speech, speech_lens)
        features_lens = features_lens.to(speech_lens.dtype)
        # encoder
        encoder_out, encoder_out_mask = self.encoder(features, features_lens)

        # cif predictor
        acoustic_embed, token_num, _, _ = self.predictor(encoder_out,
                                                         mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lens.dtype)

        # decoder
        decoder_out = self.decoder(encoder_out, encoder_out_mask,
                                   acoustic_embed, token_num)
        # decoder_out = decoder_out.log_softmax(dim=-1)
        return decoder_out, token_num


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--ali_paraformer',
                        required=True,
                        help='ali released Paraformer model path')
    parser.add_argument('--config', required=True, help='config of paraformer')
    parser.add_argument('--cmvn',
                        required=True,
                        help='cmvn file of paraformer in wenet style')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--wav', required=True, help='wav file')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    symbol_table = read_symbol_table(args.dict)
    char_dict = {v: k for k, v in symbol_table.items()}
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    mean, istd = load_cmvn(args.cmvn, is_json=True)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float())
    configs['encoder_conf']['input_size'] = 80 * 7
    encoder = SanmEncoder(global_cmvn=global_cmvn, **configs['encoder_conf'])
    configs['decoder_conf']['vocab_size'] = len(char_dict)
    configs['decoder_conf']['encoder_output_size'] = encoder.output_size()
    decoder = SanmDecoer(**configs['decoder_conf'])

    predictor = PredictorV3(**configs['predictor_conf'])

    model = Paraformer(encoder, decoder, predictor)
    load_checkpoint(model, args.ali_paraformer)
    model.eval()

    waveform, sample_rate = torchaudio.load(args.wav)
    assert sample_rate == 16000
    waveform = waveform * (1 << 15)
    waveform = waveform.to(torch.float)
    feats = kaldi.fbank(waveform,
                        num_mel_bins=80,
                        frame_length=25,
                        frame_shift=10,
                        energy_floor=0.0,
                        sample_frequency=sample_rate)
    feats = feats.unsqueeze(0)
    feats_lens = torch.tensor([feats.size(1)], dtype=torch.int64)

    out, token_nums = model(feats, feats_lens)
    print("".join([char_dict[id] for id in out.argmax(-1)[0].numpy()]))
    print(token_nums)


if __name__ == "__main__":

    main()
