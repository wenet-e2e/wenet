from typing import List, Tuple
import torch

from wenet.transformer.search import DecodeResult
from wenet.utils.mask import make_non_pad_mask, mask_finished_preds, mask_finished_scores


def paraformer_greedy_search(
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor) -> List[DecodeResult]:
    batch_size = decoder_out.shape[0]
    maxlen = decoder_out.size(1)
    topk_prob, topk_index = decoder_out.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    results = []
    # TODO(Mddct): scores, times etc
    for (i, hyp) in enumerate(topk_index.tolist()):
        r = DecodeResult(hyp[:decoder_out_lens.numpy()[i]])
        results.append(r)
    return results


def paraformer_beam_search(decoder_out: torch.Tensor,
                           decoder_out_lens: torch.Tensor,
                           beam_size: int = 10) -> List[DecodeResult]:
    mask = make_non_pad_mask(decoder_out_lens)
    indices, _ = _batch_beam_search(decoder_out, mask, beam_size=beam_size)

    best_hyps = indices[:, 0, :]
    results = []
    # TODO(Mddct): scores, times etc
    for (i, hyp) in enumerate(best_hyps.tolist()):
        r = DecodeResult(hyp[:decoder_out_lens.numpy()[i]])
        results.append(r)
    return results


def _batch_beam_search(
    logit: torch.Tensor,
    masks: torch.Tensor,
    beam_size: int = 10,
    eos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Perform batch beam search

        Args:
            logit: shape (batch_size, seq_length, vocab_size)
            masks: shape (batch_size, seq_length)
            beam_size: beam size

        Returns:
            indices: shape (batch_size, beam_size, seq_length)
            log_prob: shape (batch_size, beam_size)

        """

    batch_size, seq_length, vocab_size = logit.shape
    masks = ~masks
    # beam search
    with torch.no_grad():
        # b,t,v
        log_post = torch.nn.functional.log_softmax(logit, dim=-1)
        # b,k
        log_prob, indices = log_post[:, 0, :].topk(beam_size, sorted=True)
        end_flag = torch.eq(masks[:, 0], 1).view(-1, 1)
        # mask predictor and scores if end
        log_prob = mask_finished_scores(log_prob, end_flag)
        indices = mask_finished_preds(indices, end_flag, eos)
        # b,k,1
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            # b,v
            scores = mask_finished_scores(log_post[:, i, :], end_flag)
            # b,v -> b,k,v
            topk_scores = scores.unsqueeze(1).repeat(1, beam_size, 1)
            # b,k,1 + b,k,v -> b,k,v
            top_k_logp = log_prob.unsqueeze(-1) + topk_scores

            # b,k,v -> b,k*v -> b,k
            log_prob, top_k_index = top_k_logp.view(batch_size,
                                                    -1).topk(beam_size,
                                                             sorted=True)

            index = mask_finished_preds(top_k_index, end_flag, eos)

            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

            end_flag = torch.eq(masks[:, i], 1).view(-1, 1)

        indices = torch.fmod(indices, vocab_size)

    return indices, log_prob
