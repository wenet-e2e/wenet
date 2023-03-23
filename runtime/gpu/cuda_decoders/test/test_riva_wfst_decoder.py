import numpy as np
import time
import torch
import os
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig

class RivaWFSTDecoder:
    def __init__(self, vocab_size, tlg_dir, beam_size=8.0):
        config = BatchedMappedDecoderCudaConfig()
        config.online_opts.lattice_postprocessor_opts.acoustic_scale=10.0
        config.n_input_per_chunk = 50
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = beam_size
        config.online_opts.decoder_opts.max_active = 7000
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 800

        config.online_opts.num_channels = 800
        config.online_opts.frame_shift_seconds = 0.04

        config.online_opts.lattice_postprocessor_opts.lm_scale = 5.0
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0

        self.decoder = BatchedMappedDecoderCuda(
            config, os.path.join(tlg_dir, "TLG.fst"), os.path.join(tlg_dir, "words.txt"), vocab_size
        )

    def decode(self, logits, length):
        padded_sequence = logits.contiguous()
        sequence_lengths_tensor = length.to(torch.long).to('cpu').contiguous()
        results = self.decoder.decode(padded_sequence, sequence_lengths_tensor)
        return results

    def get_nbest_list(self, results, nbest=1):
        assert nbest == 1, "Only support nbest=1 for now"
        total_hyps = []
        for sent in results:
            hyp = [word[0] for word in sent]
            hyp_zh = "".join(hyp)
            nbest_list = [hyp_zh] # TODO: add real nbest
            total_hyps.append(nbest_list)
        return total_hyps
       

if __name__ == "__main__":
    lang_dir = "../data/lang_test" # TLG.fst, words.txt
    data = np.load('./data/input.npz')

    beam_size = 10
    batch_size = 50
    counts = 1

    # ctc_log_probs [1,103,4233]
    ctc_log_probs = torch.from_numpy(data['ctc_log_probs'])  
    # ctc_log_probs , [batch_size,T,vocab_size ]
    ctc_log_probs = ctc_log_probs.repeat(batch_size,1,1)
    encoder_out_lens = torch.from_numpy(data['encoder_out_lens'])   # encoder_out_lens single element 103
    encoder_out_lens = encoder_out_lens.repeat(batch_size)          # [batch_size]
    ctc_log_probs = ctc_log_probs.contiguous().cuda()

    vocab_size = ctc_log_probs.shape[2]
    riva_decoder = RivaWFSTDecoder(vocab_size, lang_dir, beam_size)

    decode_start = time.perf_counter()
    for i in range(counts):
        print("ctc_log_probs.shape:", ctc_log_probs.shape)
        results = riva_decoder.decode(ctc_log_probs, encoder_out_lens)
        total_hyps = riva_decoder.get_nbest_list(results)
        print(total_hyps)
    decode_end = time.perf_counter() - decode_start
    print(f"Decode {ctc_log_probs.shape[0] * counts} sentences, cost {decode_end} seconds")
