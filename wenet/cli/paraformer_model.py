import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.paraformer.search import paraformer_beam_search, paraformer_greedy_search
from wenet.utils.file_utils import read_symbol_table


class Paraformer:

    def __init__(self, model_dir: str) -> None:

        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        symbol_table = read_symbol_table(units_path)
        self.char_dict = {v: k for k, v in symbol_table.items()}
        self.eos = 2

    def transcribe(self, audio_file: str):
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=16000)
        feats = feats.unsqueeze(0)
        feats_lens = torch.tensor([feats.size(1)], dtype=torch.int64)

        decoder_out, token_num = self.model.forward_paraformer(
            feats, feats_lens)

        results = paraformer_greedy_search(decoder_out, token_num)
        hyp = [self.char_dict[x] for x in results[0].tokens]

        # TODO(Mddct): deal with '@@'
        result = ''.join(hyp)
        return result
