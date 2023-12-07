import os
import torch
from typing import List
from riva.asrlib.decoder.python_decoder import (
    BatchedMappedOnlineDecoderCuda,
    BatchedMappedDecoderCudaConfig,
)
# from frame_reducer import FrameReducer


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
    topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)
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


class RivaWFSTOnlineDecoder:

    def __init__(self, vocab_size, tlg_dir, config_dict):
        config = BatchedMappedDecoderCudaConfig()
        config.online_opts.decoder_opts.lattice_beam = config_dict[
            "lattice_beam"]
        config.online_opts.lattice_postprocessor_opts.acoustic_scale = config_dict[
            "acoustic_scale"]  # noqa
        config.n_input_per_chunk = config_dict["n_input_per_chunk"]
        config.online_opts.decoder_opts.default_beam = config_dict[
            "default_beam"]
        config.online_opts.decoder_opts.max_active = config_dict["max_active"]
        config.online_opts.determinize_lattice = config_dict[
            "determinize_lattice"]
        config.online_opts.max_batch_size = config_dict["max_batch_size"]
        config.online_opts.num_channels = config_dict["num_channels"]
        config.online_opts.frame_shift_seconds = config_dict[
            "frame_shift_seconds"]
        config.online_opts.lattice_postprocessor_opts.lm_scale = config_dict[
            "lm_scale"]
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = config_dict[
            "word_ins_penalty"]  # noqa

        config.online_opts.decoder_opts.ntokens_pre_allocated = 10_000_000
        config.online_opts.num_decoder_copy_threads = 2
        config.online_opts.num_post_processing_worker_threads = 4
        config.online_opts.use_final_probs = False

        self.decoder = BatchedMappedOnlineDecoderCuda(
            config.online_opts,
            os.path.join(tlg_dir, "TLG.fst"),
            os.path.join(tlg_dir, "words.txt"),
            vocab_size,
        )
        self.word_id_to_word_str = load_word_symbols(
            os.path.join(tlg_dir, "words.txt"))
        # self.frame_reducer = FrameReducer(0.98)

    def decode_batch(
        self,
        ctc_log_probs,
        encoder_out_lens,
        corr_ids,
        is_first_chunk_list,
        is_last_chunk_list,
        sep_symbol="",
    ):
        # ctc_log_probs, encoder_out_lens = self.frame_reducer(ctc_log_probs,
        # encoder_out_lens, ctc_log_probs)
        # if ctc_log_probs.shape[1] == 0:
        #    ctc_log_probs = torch.zeros((ctc_log_probs.shape[0],
        #                                 1, ctc_log_probs.shape[2]),
        #                                 dtype=ctc_log_probs.dtype,
        #                                 device=ctc_log_probs.device)
        ctc_log_probs = ctc_log_probs.to(torch.float32).contiguous()
        # log_probs_list = [t for t in torch.unbind(ctc_log_probs, dim=0)]
        log_probs_list = []
        for i, ctc_log_prob in enumerate(ctc_log_probs):
            log_probs_list.append(ctc_log_prob[:encoder_out_lens[i]])
        _, hypos = self.decoder.decode_batch(corr_ids, log_probs_list,
                                             is_first_chunk_list,
                                             is_last_chunk_list)
        total_hyps = []
        for sent in hypos:
            hyp = sep_symbol.join(self.word_id_to_word_str[word]
                                  for word in sent.words)
            total_hyps.append(hyp)
        return total_hyps

    def initialize_sequence(self, corr_id):
        success = self.decoder.try_init_corr_id(corr_id)
        return success
