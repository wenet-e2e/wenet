# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet) and
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from wenet.transformer.ctc import CTC
from wenet.cif.cif_decoder import CIFDecoderSAN
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.cif.predictor import MAELoss
from wenet.cif.utils import make_pad_mask
from wenet.cif.search.beam_search import Hypothesis
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy)


class ASRCIFModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            vocab_size: int,
            encoder: TransformerEncoder,
            decoder: CIFDecoderSAN,
            ctc: CTC,
            predictor,
            ctc_weight: float = 0.5,
            predictor_weight: float = 1.0,
            predictor_bias: int = 0,
            ignore_id: int = IGNORE_ID,
            reverse_weight: float = 0.0,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= predictor_weight <= 1.0, predictor_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.criterion_pre = MAELoss(normalize_length=length_normalized_loss)
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, loss_pre = self._calc_att_loss(encoder_out,
                                                              encoder_out_lens,
                                                              text,
                                                              text_lengths)
        else:
            # loss_att = None
            # loss_pre = None
            loss_att: torch.Tensor = torch.tensor(0)
            loss_pre: torch.Tensor = torch.tensor(0)

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att + self.predictor_weight * loss_pre
        # elif loss_att is None:
        elif loss_att == torch.tensor(0):
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + \
                (1 - self.ctc_weight) * loss_att + \
                self.predictor_weight * loss_pre
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc,
                "loss_pre": loss_pre}

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        encoder_out_mask = (~make_pad_mask(
            encoder_out_lens,
            maxlen=encoder_out.size(1))[:, None, :]).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = \
            self.predictor(encoder_out, ys_pad, encoder_out_mask,
                           ignore_id=self.ignore_id)
        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre: torch.Tensor = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length)

        return loss_att, acc_att, loss_pre

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))
            [:, None, :]).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = \
            self.predictor(
                encoder_out, None,
                encoder_out_mask,
                ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens,
                                   sematic_embeds, ys_pad_lens):

        decoder_outs = self.decoder(encoder_out, encoder_out_lens,
                                    sematic_embeds, ys_pad_lens)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(self):
        raise NotImplementedError

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
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
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen).to(topk_index.device)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
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
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
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
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        return hyps[0]

    def cif_greedy_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2. Predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = \
            predictor_outs[0], predictor_outs[1], \
            predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return torch.tensor([]), torch.tensor([])
        # 2. Decoder forward
        decoder_outs = self.cal_decoder_with_predictor(encoder_out,
                                                       encoder_out_lens,
                                                       pre_acoustic_embeds,
                                                       pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        hyps = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = encoder_out[i, :encoder_out_lens[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            yseq = am_scores.argmax(dim=-1)
            score = am_scores.max(dim=-1)[0]
            score = torch.sum(score, dim=-1)
            # pad with mask tokens to ensure compatibility with sos/eos tokens
            yseq = torch.tensor(
                [self.sos] + yseq.tolist() + [self.eos], device=yseq.device
            )
            nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get hyps
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id and unk id, which is assumed to be 0
                # and 1
                token_int = list(filter(lambda x: x != 0 and x != 1, token_int))
                hyps.append(token_int)
        return hyps

    def cif_beam_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_search: torch.nn.Module = None,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
            beam_search (torch.nn.Moudle): beam search module
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2. Predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = \
            predictor_outs[0], predictor_outs[1], \
            predictor_outs[2], predictor_outs[3]
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return torch.tensor([]), torch.tensor([])
        # 2. Decoder forward
        decoder_outs = self.cal_decoder_with_predictor(encoder_out,
                                                       encoder_out_lens,
                                                       pre_acoustic_embeds,
                                                       pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        hyps = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = encoder_out[i, :encoder_out_lens[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if beam_search is not None:
                nbest_hyps = beam_search(x=x, am_scores=am_scores)
                nbest_hyps = nbest_hyps[: 1]
            else:
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos
                # tokens
                yseq = torch.tensor(
                    [self.sos] + yseq.tolist() + [self.eos], device=yseq.device
                )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get hyps
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id and unk id, which is assumed to be 0
                # and 1
                token_int = list(filter(lambda x: x != 0 and x != 1, token_int))
                hyps.append(token_int)
        return hyps
