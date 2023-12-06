import os
import torch
from typing import List
from riva.asrlib.decoder.python_decoder import (BatchedMappedDecoderCuda,
                                                BatchedMappedDecoderCudaConfig)
from frame_reducer import FrameReducer


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.
    See description of make_non_pad_mask.
    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.
    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def remove_duplicates_and_blank(hyp: List[int],
                                eos: int,
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id and hyp[cur] != eos:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def ctc_greedy_search(ctc_probs, encoder_out_lens, vocabulary, blank_id, eos):
    batch_size, maxlen = ctc_probs.size()[:2]
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.cpu().masked_fill_(mask, eos)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    hyps = [remove_duplicates_and_blank(hyp, eos, blank_id) for hyp in hyps]
    total_hyps = []
    for hyp in hyps:
        total_hyps.append("".join([vocabulary[i] for i in hyp]))
    return total_hyps


def load_word_symbols(path):
    word_id_to_word_str = {}
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            word_str, word_id = line.rstrip().split()
            word_id_to_word_str[int(word_id)] = word_str
    return word_id_to_word_str


class RivaWFSTDecoder:

    def __init__(self, vocab_size, tlg_dir, config_dict, nbest=10):
        config = BatchedMappedDecoderCudaConfig()
        config.online_opts.decoder_opts.lattice_beam = config_dict[
            'lattice_beam']
        config.online_opts.lattice_postprocessor_opts.acoustic_scale = config_dict[
            'acoustic_scale']  # noqa
        config.n_input_per_chunk = config_dict['n_input_per_chunk']
        config.online_opts.decoder_opts.default_beam = config_dict[
            'default_beam']
        config.online_opts.decoder_opts.max_active = config_dict['max_active']
        config.online_opts.determinize_lattice = config_dict[
            'determinize_lattice']
        config.online_opts.max_batch_size = config_dict['max_batch_size']
        config.online_opts.num_channels = config_dict['num_channels']
        config.online_opts.frame_shift_seconds = config_dict[
            'frame_shift_seconds']
        config.online_opts.lattice_postprocessor_opts.lm_scale = config_dict[
            'lm_scale']
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = config_dict[
            'word_ins_penalty']  # noqa

        config.online_opts.num_decoder_copy_threads = 2
        config.online_opts.num_post_processing_worker_threads = 4
        config.online_opts.lattice_postprocessor_opts.nbest = nbest

        # config.online_opts.decoder_opts.blank_penalty = -5.0

        self.decoder = BatchedMappedDecoderCuda(
            config, os.path.join(tlg_dir, "TLG.fst"),
            os.path.join(tlg_dir, "words.txt"), vocab_size)
        self.word_id_to_word_str = load_word_symbols(
            os.path.join(tlg_dir, "words.txt"))
        self.nbest = nbest
        self.vocab_size = vocab_size
        self.frame_reducer = FrameReducer(0.98)

    def decode_nbest(self, logits, length):
        logits, length = self.frame_reducer(logits, length.cuda(), logits)
        logits = logits.to(torch.float32).contiguous()
        sequence_lengths_tensor = length.to(torch.long).to('cpu').contiguous()
        results = self.decoder.decode_nbest(logits, sequence_lengths_tensor)
        total_hyps, total_hyps_id, total_scores = [], [], []
        max_hyp_len = 3  # [sos, 0, eos]
        for nbest_sentences in results:
            nbest_list, nbest_id_list, nbest_scores = [], [], []
            for sent in nbest_sentences:
                # subtract 1 to get the label id,
                # since fst decoder adds 1 to the label id
                hyp_ids = [label - 1 for label in sent.ilabels]
                # padding for hyps_pad_sos_eos
                new_hyp = [self.vocab_size - 1] + remove_duplicates_and_blank(
                    hyp_ids, eos=self.vocab_size - 1,
                    blank_id=0) + [self.vocab_size - 1]  # noqa
                max_hyp_len = max(max_hyp_len, len(new_hyp))
                nbest_id_list.append(new_hyp)

                hyp = "".join(self.word_id_to_word_str[word]
                              for word in sent.words if word != 0)
                nbest_list.append(hyp)
                nbest_scores.append(sent.score)
            nbest_list += [""] * (self.nbest - len(nbest_list))
            total_hyps.append(nbest_list)
            nbest_id_list += [[self.vocab_size - 1, 0, self.vocab_size - 1]
                              ] * (self.nbest - len(nbest_id_list))  # noqa
            total_hyps_id.append(nbest_id_list)
            nbest_scores += [0.0] * (self.nbest - len(nbest_scores))
            total_scores.append(nbest_scores)
        return total_hyps, total_hyps_id, total_scores, max_hyp_len

    def decode_mbr(self, logits, length):
        logits, length = self.frame_reducer(logits, length.cuda(), logits)
        # logits[:,:,0] -= 2.0
        logits = logits.to(torch.float32).contiguous()
        sequence_lengths_tensor = length.to(torch.long).to('cpu').contiguous()
        results = self.decoder.decode_mbr(logits, sequence_lengths_tensor)
        total_hyps = []
        for sent in results:
            hyp = [word[0] for word in sent]
            hyp_zh = "".join(hyp)
            total_hyps.append(hyp_zh)
        return total_hyps
