from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from wenet.transducer.predictor import PredictorBase
from wenet.transducer.search.greedy_search import basic_greedy_search
from wenet.transducer.search.prefix_beam_search import PrefixBeamSearch
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_blank, add_sos_eos,
                                reverse_pad_list, TORCH_NPU_AVAILABLE)


class Transducer(ASRModel):
    """Transducer-ctc-attention hybrid Encoder-Predictor-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        attention_decoder: Optional[Union[TransformerDecoder,
                                          BiTransformerDecoder]] = None,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        transducer_weight: float = 1.0,
        attention_weight: float = 0.0,
        enable_k2: bool = False,
        delay_penalty: float = 0.0,
        warmup_steps: float = 25000,
        lm_only_scale: float = 0.25,
        am_only_scale: float = 0.0,
        special_tokens: dict = None,
    ) -> None:
        assert attention_weight + ctc_weight + transducer_weight == 1.0
        super().__init__(vocab_size,
                         encoder,
                         attention_decoder,
                         ctc,
                         ctc_weight,
                         ignore_id,
                         reverse_weight,
                         lsm_weight,
                         length_normalized_loss,
                         special_tokens=special_tokens)

        self.blank = blank
        self.transducer_weight = transducer_weight
        self.attention_decoder_weight = 1 - self.transducer_weight - self.ctc_weight

        self.predictor = predictor
        self.joint = joint
        self.bs = None

        # k2 rnnt loss
        self.enable_k2 = enable_k2
        self.delay_penalty = delay_penalty
        if delay_penalty != 0.0:
            assert self.enable_k2 is True
        self.lm_only_scale = lm_only_scale
        self.am_only_scale = am_only_scale
        self.warmup_steps = warmup_steps
        self.simple_am_proj: Optional[nn.Linear] = None
        self.simple_lm_proj: Optional[nn.Linear] = None
        if self.enable_k2:
            self.simple_am_proj = torch.nn.Linear(self.encoder.output_size(),
                                                  vocab_size)
            self.simple_lm_proj = torch.nn.Linear(self.predictor.output_size(),
                                                  vocab_size)

        # Note(Mddct): decoder also means predictor in transducer,
        # but here decoder is attention decoder
        del self.criterion_att
        if attention_decoder is not None:
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss
        """
        self.device = device
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        steps = batch.get('steps', 0)
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # compute_loss
        loss_rnnt = self._compute_loss(encoder_out,
                                       encoder_out_lens,
                                       encoder_mask,
                                       text,
                                       text_lengths,
                                       steps=steps)

        loss = self.transducer_weight * loss_rnnt
        # optional attention decoder
        loss_att: Optional[torch.Tensor] = None
        if self.attention_decoder_weight != 0.0 and self.decoder is not None:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            acc_att = None

        # optional ctc
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text,
                                   text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc.sum()
        if loss_att is not None:
            loss = loss + self.attention_decoder_weight * loss_att.sum()
        # NOTE: 'loss' must be in dict
        return {
            'loss': loss,
            'loss_att': loss_att,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
            'th_accuracy': acc_att,
        }

    def init_bs(self):
        if self.bs is None:
            self.bs = PrefixBeamSearch(self.encoder, self.predictor,
                                       self.joint, self.ctc, self.blank)

    def _cal_transducer_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps_lens: torch.Tensor,
        hyps_pad: torch.Tensor,
    ):
        # ignore id -> blank, add blank at head
        hyps_pad_blank = add_blank(hyps_pad, self.blank, self.ignore_id)
        xs_in_lens = encoder_mask.squeeze(1).sum(1).int()

        # 1. Forward predictor
        predictor_out = self.predictor(hyps_pad_blank)
        # 2. Forward joint
        joint_out = self.joint(encoder_out, predictor_out)
        rnnt_text = hyps_pad.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                rnnt_text).to(torch.int32)
        # 3. Compute transducer loss
        loss_td = torchaudio.functional.rnnt_loss(joint_out,
                                                  rnnt_text,
                                                  xs_in_lens,
                                                  hyps_lens.int(),
                                                  blank=self.blank,
                                                  reduction='none')
        return loss_td * -1

    def _cal_attn_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps_pad: torch.Tensor,
        hyps_lens: torch.Tensor,
    ):
        # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad

        # td_score = loss_td * -1
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            self.reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        return decoder_out, r_decoder_out

    def beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
    ):
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight in transducer
                prefix beam search.
                final_prob = ctc_weight * ctc_prob + transducer_weight * transducer_prob
            transducer_weight (float): transducer probability weight in
                prefix beam search
        Returns:
            List[List[int]]: best path result

        """
        self.init_bs()
        beam, _ = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            simulate_streaming,
            ctc_weight,
            transducer_weight,
        )
        return beam[0].hyp[1:], beam[0].score

    def transducer_attention_rescoring(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
            reverse_weight: float = 0.0,
            ctc_weight: float = 0.0,
            attn_weight: float = 0.0,
            transducer_weight: float = 0.0,
            search_ctc_weight: float = 1.0,
            search_transducer_weight: float = 0.0,
            beam_search_type: str = 'transducer') -> List[List[int]]:
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight using in rescoring.
                rescore_prob = ctc_weight * ctc_prob +
                               transducer_weight * (transducer_loss * -1) +
                               attn_weight * attn_prob
            attn_weight (float): attn probability weight using in rescoring.
            transducer_weight (float): transducer probability weight using in
                rescoring
            search_ctc_weight (float): ctc weight using
                               in rnnt beam search (seeing in self.beam_search)
            search_transducer_weight (float): transducer weight using
                               in rnnt beam search (seeing in self.beam_search)
        Returns:
            List[List[int]]: best path result

        """

        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        self.init_bs()
        if beam_search_type == 'transducer':
            beam, encoder_out = self.bs.prefix_beam_search(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                beam_size=beam_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                ctc_weight=search_ctc_weight,
                transducer_weight=search_transducer_weight,
            )
            beam_score = [s.score for s in beam]
            hyps = [s.hyp[1:] for s in beam]

        elif beam_search_type == 'ctc':
            hyps, encoder_out = self._ctc_prefix_beam_search(
                speech,
                speech_lengths,
                beam_size=beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            beam_score = [hyp[1] for hyp in hyps]
            hyps = [hyp[0] for hyp in hyps]
        assert len(hyps) == beam_size

        # build hyps and encoder output
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)

        # 2.1 calculate transducer score
        td_score = self._cal_transducer_score(
            encoder_out,
            encoder_mask,
            hyps_lens,
            hyps_pad,
        )
        # 2.2 calculate attention score
        decoder_out, r_decoder_out = self._cal_attn_score(
            encoder_out,
            encoder_mask,
            hyps_pad,
            hyps_lens,
        )

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp)][self.eos]
            td_s = td_score[i]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    r_score += r_decoder_out[i][len(hyp) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp)][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score = score * attn_weight + \
                beam_score[i] * ctc_weight + \
                td_s * transducer_weight
            if score > best_score:
                best_score = score
                best_index = i

        return hyps[best_index], best_score

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """ greedy search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        # TODO(Mddct): batch decode
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # TODO(Mddct): forward chunk by chunk
        _ = simulate_streaming
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = basic_greedy_search(self,
                                   encoder_out,
                                   encoder_out_lens,
                                   n_steps=n_steps)

        return hyps

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def forward_predictor_step(
            self, xs: torch.Tensor, cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(cache) == 2
        # fake padding
        padding = torch.zeros(1, 1)
        return self.predictor.forward_step(xs, padding, cache)

    @torch.jit.export
    def forward_joint_step(self, enc_out: torch.Tensor,
                           pred_out: torch.Tensor) -> torch.Tensor:
        return self.joint(enc_out, pred_out)

    @torch.jit.export
    def forward_predictor_init_state(self) -> List[torch.Tensor]:
        return self.predictor.init_state(1, device=torch.device("cpu"))

    def _compute_loss(self,
                      encoder_out: torch.Tensor,
                      encoder_out_lens: torch.Tensor,
                      encoder_mask: torch.Tensor,
                      text: torch.Tensor,
                      text_lengths: torch.Tensor,
                      steps: int = 0) -> torch.Tensor:
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        # predictor
        predictor_out = self.predictor(ys_in_pad)
        if self.simple_lm_proj is None and self.simple_am_proj is None:
            # joint
            joint_out = self.joint(encoder_out, predictor_out)
            # NOTE(Mddct): some loss implementation require pad valid is zero
            # torch.int32 rnnt_loss required
            rnnt_text = text.to(torch.int64)
            rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                    rnnt_text).to(torch.int32)
            rnnt_text_lengths = text_lengths.to(torch.int32)
            encoder_out_lens = encoder_out_lens.to(torch.int32)
            loss = torchaudio.functional.rnnt_loss(joint_out,
                                                   rnnt_text,
                                                   encoder_out_lens,
                                                   rnnt_text_lengths,
                                                   blank=self.blank,
                                                   reduction="mean")
        else:
            try:
                import k2
            except ImportError:
                print('Error: k2 is not installed')
            delay_penalty = self.delay_penalty
            if steps < 2 * self.warmup_steps:
                delay_penalty = 0.00
            ys_in_pad = ys_in_pad.type(torch.int64)
            boundary = torch.zeros((encoder_out.size(0), 4),
                                   dtype=torch.int64,
                                   device=encoder_out.device)
            boundary[:, 3] = encoder_mask.squeeze(1).sum(1)
            boundary[:, 2] = text_lengths

            rnnt_text = torch.where(text == self.ignore_id, 0, text)
            lm = self.simple_lm_proj(predictor_out)
            am = self.simple_am_proj(encoder_out)
            amp_autocast = torch.cuda.amp.autocast
            if "npu" in self.device.__str__() and TORCH_NPU_AVAILABLE:
                amp_autocast = torch.npu.amp.autocast
            with amp_autocast(enabled=False):
                simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                    lm=lm.float(),
                    am=am.float(),
                    symbols=rnnt_text,
                    termination_symbol=self.blank,
                    lm_only_scale=self.lm_only_scale,
                    am_only_scale=self.am_only_scale,
                    boundary=boundary,
                    reduction="sum",
                    return_grad=True,
                    delay_penalty=delay_penalty,
                )
            # ranges : [B, T, prune_range]
            ranges = k2.get_rnnt_prune_ranges(
                px_grad=px_grad,
                py_grad=py_grad,
                boundary=boundary,
                s_range=5,
            )
            am_pruned, lm_pruned = k2.do_rnnt_pruning(
                am=self.joint.enc_ffn(encoder_out),
                lm=self.joint.pred_ffn(predictor_out),
                ranges=ranges,
            )
            logits = self.joint(
                am_pruned,
                lm_pruned,
                pre_project=False,
            )
            with amp_autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=rnnt_text,
                    ranges=ranges,
                    termination_symbol=self.blank,
                    boundary=boundary,
                    reduction="sum",
                    delay_penalty=delay_penalty,
                )
            simple_loss_scale = 0.5
            if steps < self.warmup_steps:
                simple_loss_scale = (1.0 - (steps / self.warmup_steps) *
                                     (1.0 - simple_loss_scale))
            pruned_loss_scale = 1.0
            if steps < self.warmup_steps:
                pruned_loss_scale = 0.1 + 0.9 * (steps / self.warmup_steps)
            loss = (simple_loss_scale * simple_loss +
                    pruned_loss_scale * pruned_loss)
            loss = loss / encoder_out.size(0)
        return loss
