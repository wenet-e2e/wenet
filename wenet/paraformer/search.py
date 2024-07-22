import math
from typing import Any, List, Optional, Tuple, Union
import torch

from wenet.transformer.search import DecodeResult
from wenet.utils.mask import (make_non_pad_mask, mask_finished_preds,
                              mask_finished_scores)


def _isChinese(ch: str):
    if '\u4e00' <= ch <= '\u9fff' or '\u0030' <= ch <= '\u0039' or ch == '@':
        return True
    return False


def _isAllChinese(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        cur = cur.replace('<unk>', '')
        cur = cur.replace('<OOV>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if _isChinese(ch) is False:
            return False
    return True


def _isAllAlpha(word: Union[List[Any], str]):
    word_lists = []
    for i in word:
        cur = i.replace(' ', '')
        cur = cur.replace('</s>', '')
        cur = cur.replace('<s>', '')
        cur = cur.replace('<unk>', '')
        cur = cur.replace('<OOV>', '')
        word_lists.append(cur)

    if len(word_lists) == 0:
        return False

    for ch in word_lists:
        if ch.isalpha() is False and ch != "'":
            return False
        elif ch.isalpha() is True and _isChinese(ch) is True:
            return False

    return True


def paraformer_beautify_result(tokens: List[str]) -> str:
    middle_lists = []
    word_lists = []
    word_item = ''

    # wash words lists
    for token in tokens:
        if token in ['<sos>', '<eos>', '<blank>']:
            continue
        else:
            middle_lists.append(token)

    # all chinese characters
    if _isAllChinese(middle_lists):
        for _, ch in enumerate(middle_lists):
            word_lists.append(ch.replace(' ', ''))

    # all alpha characters
    elif _isAllAlpha(middle_lists):
        for _, ch in enumerate(middle_lists):
            word = ''
            if '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''

    # mix characters
    else:
        alpha_blank = False
        for _, ch in enumerate(middle_lists):
            word = ''
            if _isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
            elif '@@' in ch:
                word = ch.replace('@@', '')
                word_item += word
                alpha_blank = False
            elif _isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(' ')
                word_item = ''
                alpha_blank = True
            else:
                word_lists.append(ch)
                alpha_blank = False
    return ''.join(word_lists).strip()


def gen_timestamps_from_peak(cif_peaks: List[int],
                             num_frames: int,
                             frame_rate=0.02):
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 14
    force_time_shift = -0.5
    fire_place = [peak + force_time_shift for peak in cif_peaks]
    times = []
    for i in range(len(fire_place) - 1):
        if MAX_TOKEN_DURATION < 0 or fire_place[
                i + 1] - fire_place[i] <= MAX_TOKEN_DURATION:
            times.append(
                [fire_place[i] * frame_rate, fire_place[i + 1] * frame_rate])
        else:
            split = fire_place[i] + MAX_TOKEN_DURATION
            times.append([fire_place[i] * frame_rate, split * frame_rate])
    if len(times) > 0:
        if num_frames - fire_place[-1] > START_END_THRESHOLD:
            end = (num_frames + fire_place[-1]) * 0.5
            times[-1][1] = end * frame_rate
            times.append([end * frame_rate, num_frames * frame_rate])
        else:
            times[-1][1] = num_frames * frame_rate
    return times


def paraformer_greedy_search(
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        cif_peaks: Optional[torch.Tensor] = None) -> List[DecodeResult]:
    batch_size = decoder_out.shape[0]
    maxlen = decoder_out.size(1)
    topk_prob, topk_index = decoder_out.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_prob = topk_prob.view(batch_size, maxlen)
    results: List[DecodeResult] = []
    topk_index = topk_index.cpu().tolist()
    topk_prob = topk_prob.cpu().tolist()
    decoder_out_lens = decoder_out_lens.cpu().numpy()
    for (i, hyp) in enumerate(topk_index):
        confidence = 0.0
        tokens_confidence = []
        lens = decoder_out_lens[i]
        for logp in topk_prob[i][:lens]:
            tokens_confidence.append(math.exp(logp))
            confidence += logp
        r = DecodeResult(hyp[:lens],
                         tokens_confidence=tokens_confidence,
                         confidence=math.exp(confidence / lens))
        results.append(r)

    if cif_peaks is not None:
        for (b, peaks) in enumerate(cif_peaks):
            result = results[b]
            times = []
            n_token = 0
            for (i, peak) in enumerate(peaks):
                if n_token >= len(result.tokens):
                    break
                if peak > 1 - 1e-4:
                    times.append(i)
                    n_token += 1
            result.times = times
            assert len(result.times) == len(result.tokens)
    return results


def paraformer_beam_search(decoder_out: torch.Tensor,
                           decoder_out_lens: torch.Tensor,
                           beam_size: int = 10,
                           eos: int = -1) -> List[DecodeResult]:
    mask = make_non_pad_mask(decoder_out_lens)
    indices, _ = _batch_beam_search(decoder_out,
                                    mask,
                                    beam_size=beam_size,
                                    eos=eos)

    best_hyps = indices[:, 0, :].cpu()
    decoder_out_lens = decoder_out_lens.cpu()
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
