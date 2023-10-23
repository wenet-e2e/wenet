# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import (add_sos_eos, log_add, reverse_pad_list)
from wenet.utils.ctc_utils import remove_duplicates_and_blank
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[Tuple[float, float]] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores


class PrefixScore:
    """ For CTC prefix beam search """

    def __init__(self, s=float('-inf'), ns=float('-inf')):

        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score

    def score(self):
        return log_add(self.s, self.ns)


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp))
        results.append(r)
    return results


def ctc_prefix_beam_search(ctc_probs: torch.Tensor, ctc_lens: torch.Tensor,
                           beam_size: int) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    batch_size = ctc_probs.shape[0]
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(), PrefixScore(0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for u in top_k_index:
                u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == 0:  # blank
                        next_hyps[prefix].s = log_add(
                            next_hyps[prefix].s,
                            prefix_score.score() + prob)
                    elif u == last:
                        #  Update *uu -> *u;
                        next_hyps[prefix].ns = log_add(next_hyps[prefix].ns,
                                                       prefix_score.ns + prob)
                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u, )
                        next_hyps[n_prefix].ns = log_add(
                            next_hyps[n_prefix].ns, prefix_score.s + prob)
                    else:
                        n_prefix = prefix + (u, )
                        next_hyps[n_prefix].ns = log_add(
                            next_hyps[n_prefix].ns,
                            prefix_score.score() + prob)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: x[1].score(),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]

        nbest = [y[0] for y in cur_hyps]
        nbest_scores = [y[1].score() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        results.append(
            DecodeResult(tokens=best,
                         score=best_score,
                         nbest=nbest,
                         nbest_scores=nbest_scores))
    return results


def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    beam_size: int = 10,
) -> List[DecodeResult]:
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
        running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
    encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
        running_size, 1, maxlen)  # (B*N, 1, max_len)

    hyps = torch.ones([running_size, 1], dtype=torch.long,
                      device=device).fill_(model.sos)  # (B*N, 1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                          dtype=torch.float)
    scores = scores.to(device).repeat([batch_size
                                       ]).unsqueeze(1).to(device)  # (B*N, 1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    cache: Optional[List[torch.Tensor]] = None
    # 2. Decoder forward step by step
    for i in range(1, maxlen + 1):
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            break
        # 2.1 Forward decoder step
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
            running_size, 1, 1).to(device)  # (B*N, i, i)
        # logp: (B*N, vocab)
        logp, cache = model.decoder.forward_one_step(encoder_out, encoder_mask,
                                                     hyps, hyps_mask, cache)
        # 2.2 First beam prune: select topk best prob at current time
        top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        # 2.3 Second beam prune: select topk score with history
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        # Update cache to be consistent with new topk scores / hyps
        cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
        base_cache_index = (torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
        cache_index = base_cache_index + cache_index
        cache = [
            torch.index_select(c, dim=0, index=cache_index) for c in cache
        ]
        scores = scores.view(-1, 1)  # (B*N, 1)
        # 2.4. Compute base index in top_k_index,
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
        # then find offset_k_index in top_k_index
        base_k_index = torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size])  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

        # 2.5 Update best hyps
        best_k_pred = torch.index_select(top_k_index.view(-1),
                                         dim=-1,
                                         index=best_k_index)  # (B*N)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(
            hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                         dim=1)  # (B*N, i+1)

        # 2.6 Update end flag
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)

    # 3. Select best of best
    scores = scores.view(batch_size, beam_size)
    # TODO: length normalization
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(
        batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, 1:]

    results = []
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        results.append(DecodeResult(hyp.tolist()))
    return results


def attention_rescoring(
    model,
    ctc_prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    ctc_weight: float = 0.0,
    reverse_weight: float = 0.0,
) -> List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    if reverse_weight > 0.0:
        # decoder should be a bitransformer decoder if reverse_weight > 0.0
        assert hasattr(model.decoder, 'right_decoder')
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = ctc_prefix_results[b].nbest
        ctc_scores = ctc_prefix_results[b].nbest_scores
        beam_size = len(hyps)
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, model.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, model.sos, model.eos,
                                  model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, model.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, model.sos, model.eos,
                                    model.ignore_id)
        decoder_out, r_decoder_out, _ = model.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp)][model.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    r_score += r_decoder_out[i][len(hyp) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp)][model.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += ctc_scores[i] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        results.append(DecodeResult(hyps[best_index], best_score))
    return results
