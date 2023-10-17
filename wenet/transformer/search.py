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
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import (add_sos_eos, log_add, reverse_pad_list)
from wenet.utils.ctc_utils import remove_duplicates_and_blank
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


def ctc_greedy_search(ctc_probs: torch.Tensor, ctc_lens: torch.Tensor):
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    return hyps, scores


def ctc_prefix_beam_search(ctc_probs: torch.Tensor, ctc_lens: torch.Tensor,
                           beam_size: int) -> List[List[List[int]]]:
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
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score,
        #                     context_state, context_score))
        cur_hyps = [(tuple(), (0.0, -float('inf'), 0, 0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb, context_state, context_score),
            # default value(-inf, -inf, 0, 0.0)
            next_hyps = defaultdict(lambda:
                                    (-float('inf'), -float('inf'), 0, 0.0))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb, c_state, c_score) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb, _, _ = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb, _, _ = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, _, _ = next_hyps[n_prefix]
                        new_c_state, new_c_score = 0, 0
                        # if context_graph is not None:
                        #     new_c_state, new_c_score = context_graph. \
                        #         find_next_state(c_state, s)
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state,
                                               c_score + new_c_score)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, _, _ = next_hyps[n_prefix]
                        new_c_state, new_c_score = 0, 0
                        # if context_graph is not None:
                        #     new_c_state, new_c_score = context_graph. \
                        #         find_next_state(c_state, s)
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state,
                                               c_score + new_c_score)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add([x[1][0], x[1][1]]) + x[1][3],
                reverse=True)
            cur_hyps = next_hyps[:beam_size]
        results.append([(y[0], log_add([y[1][0], y[1][1]]) + y[1][3])
                        for y in cur_hyps])
    return results


def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    beam_size: int = 10,
) -> torch.Tensor:
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
    return best_hyps, best_scores


def attention_rescoring(
    model,
    nbests: List[List[List[int]]],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    ctc_weight: float = 0.0,
    reverse_weight: float = 0.0,
) -> List[List[int]]:
    """
        Args:
            nbests(List[List[List[int]]]): ctc prefix beam search nbests
    """
    if reverse_weight > 0.0:
        # decoder should be a bitransformer decoder if reverse_weight > 0.0
        assert hasattr(model.decoder, 'right_decoder')
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(nbests)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = nbests[b]
        beam_size = len(hyps)
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, model.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
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
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][model.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][model.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        results.append(hyps[best_index][0])
    return results
