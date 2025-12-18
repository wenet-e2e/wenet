from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint as ckpt

from wenet.models.paraformer.attention import MultiHeadedAttentionSANM
from wenet.models.paraformer.layers import (LFR, AliParaformerEncoderLayer,
                                            SanmEncoder)
from wenet.models.transformer.asr_model import ASRModel
from wenet.models.transformer.ctc import CTC
from wenet.models.transformer.decoder import TransformerDecoder
from wenet.models.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.models.transformer.positionwise_feed_forward import \
    PositionwiseFeedForward
from wenet.models.transformer.search import DecodeResult
from wenet.utils.common import IGNORE_ID, mask_to_bias
from wenet.utils.context_graph import ContextGraph
from wenet.utils.mask import add_optional_chunk_mask, make_pad_mask


class SanmEncoderWithTp(SanmEncoder):

    def __init__(self,
                 input_size: int,
                 tp_blocks: int,
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
                 sanm_shfit: int = 0,
                 gradient_checkpointing: bool = False):
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, kernel_size, sanm_shfit,
                         gradient_checkpointing)
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer = MultiHeadedAttentionSANM
        self.tp_encoders = torch.nn.ModuleList([
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
                in_size=output_size) for _ in range(tp_blocks)
        ])
        self.tp_norm = torch.nn.LayerNorm(output_size)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate))
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb,
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later

        # sensevoice tp encoders:
        if self.gradient_checkpointing and self.training:
            xs = self.forward_tp_layers_checkpointed(xs, chunk_masks, pos_emb,
                                                     mask_pad)
        else:
            xs = self.forward_tp_layers(xs, chunk_masks, pos_emb, mask_pad)
        xs = self.tp_norm(xs)
        return xs, masks

    @torch.jit.unused
    def forward_tp_layers_checkpointed(self, xs: torch.Tensor,
                                       chunk_masks: torch.Tensor,
                                       pos_emb: torch.Tensor,
                                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.tp_encoders:
            xs, _, _, _, _ = ckpt.checkpoint(
                layer.__call__,
                xs,
                chunk_masks,
                pos_emb,
                mask_pad,
            )
        return xs

    def forward_tp_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                          pos_emb: torch.Tensor,
                          mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.tp_encoders:
            xs, _, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs


class SenseVoiceSmall(ASRModel):
    default_decode_method = "ctc_greedy_search"

    def __init__(self,
                 vocab_size: int,
                 encoder: SanmEncoderWithTp,
                 decoder: TransformerDecoder,
                 ctc: CTC,
                 ctc_weight: float = 0.5,
                 ignore_id: int = IGNORE_ID,
                 reverse_weight: float = 0,
                 lsm_weight: float = 0,
                 length_normalized_loss: bool = False,
                 special_tokens: Optional[dict] = None,
                 apply_non_blank_embedding: bool = False):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens,
                         apply_non_blank_embedding)

        assert ctc_weight != 0.0
        assert special_tokens is not None
        self.encoder = encoder
        self.decoder = decoder
        self.lfr = LFR()

        self.sos = special_tokens['<s>']
        self.eos = special_tokens['</s>']

        # hard code for sensevoice small
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), 560)

        assert self.encoder.global_cmvn is not None
        self.global_cmvn = self.encoder.global_cmvn
        self.encoder.global_cmvn = None

        self.criterion_context = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    @torch.jit.unused
    def tie_or_clone_weights(self, jit_mode: bool = True):
        pass

    @torch.jit.unused
    def forward(self, batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        speech, speech_lengths = self.lfr(speech, speech_lengths)
        speech = self.global_cmvn(speech)

        # context pattern:
        # lid emo event tn speech
        # TODO: move to dataset
        lid = batch['lid'].to(device).unsqueeze(1)  # [B,1]
        itn = batch['itn'].to(device).unsqueeze(1)  # [B,1]
        event_emo_query = torch.LongTensor([[1, 2]]).to(speech.device).repeat(
            speech.size(0), 1)  # [B,2]
        context = torch.stack([lid, event_emo_query, itn], dim=1)

        context_embed = self.embed(context)  # [B,4,D]
        speech = torch.cat((context_embed, speech), dim=1)
        speech_lengths = speech_lengths + 3 + 1

        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.sum(-1).squeeze()
        loss_ctc_speech = self.ctc(encoder_out[:4:, :, :],
                                   encoder_out_lens - 4, text[:, 4:],
                                   text_lengths - 4)

        context_logits = self.ctc.ctc_lo(encoder_out[:, :4, :])
        loss_context = self.criterion_context(context_logits, text[:, :4])

        loss_att, acc_att = None, 0
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)

        loss_ctc = loss_ctc_speech + loss_context
        loss = loss_ctc
        if loss_att is not None:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att

        # TODO: log context acc
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "loss_ctc_speech": loss_ctc_speech,
            "loss_context": loss_context,
            "th_accuracy": acc_att,
        }

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 1,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = {},
    ) -> Dict[str, List[DecodeResult]]:
        assert simulate_streaming is False
        speech, speech_lengths = self.lfr(speech, speech_lengths)
        speech = self.global_cmvn(speech)
        # context pattern
        itn = infos.get('itn', 'woitn')
        lid = infos.get('lid', 'auto')
        lid_query = self.embed(torch.LongTensor(
            [[self.lid_dict[lid] if lid in self.lid_dict else 0]]).to(speech.device)).repeat(
                speech.size(0), 1, 1
        )
        itn_query = self.embed(torch.LongTensor(
            [[self.textnorm_dict[itn] if itn in self.textnorm_dict else 15]]).to(speech.device)).repeat(
                speech.size(0), 1, 1
        )
        # hard code
        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        speech = torch.cat((lid_query, event_emo_query, itn_query, speech), dim=1)
        speech_lengths += 4
        return super().decode(
            methods, speech, speech_lengths, beam_size,
            decoding_chunk_size, num_decoding_left_chunks, ctc_weight,
            simulate_streaming, reverse_weight, context_graph, blank_id,
            blank_penalty, length_penalty, infos)
