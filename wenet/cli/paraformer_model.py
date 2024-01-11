import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.cli.hub import Hub
from wenet.paraformer.search import (gen_timestamps_from_peak,
                                     paraformer_greedy_search)
from wenet.text.paraformer_tokenizer import ParaformerTokenizer


class Paraformer:

    def __init__(self,
                 model_dir: str,
                 device: int = -1,
                 resample_rate: int = 16000) -> None:

        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        self.resample_rate = resample_rate
        if device >= 0:
            device = 'cuda:{}'.format(device)
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.tokenizer = ParaformerTokenizer(symbol_table=units_path)

    def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float).to(self.device)
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        feats = feats.unsqueeze(0)
        feats_lens = torch.tensor([feats.size(1)],
                                  dtype=torch.int64,
                                  device=feats.device)

        decoder_out, token_num, tp_alphas = self.model.forward_paraformer(
            feats, feats_lens)
        cif_peaks = self.model.forward_cif_peaks(tp_alphas, token_num)
        res = paraformer_greedy_search(decoder_out, token_num, cif_peaks)[0]
        result = {}
        result['confidence'] = res.confidence
        result['text'] = self.tokenizer.detokenize(res.tokens)[0]
        if tokens_info:
            tokens_info = []
            times = gen_timestamps_from_peak(res.times,
                                             num_frames=tp_alphas.size(1),
                                             frame_rate=0.02)

            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.tokenizer.char_dict[x],
                    'start': times[i][0],
                    'end': times[i][1],
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info

        return result

    def align(self, audio_file: str, label: str) -> dict:
        raise NotImplementedError("Align is currently not supported")


def load_model(model_dir: str = None, gpu: int = -1) -> Paraformer:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang('paraformer')
    return Paraformer(model_dir, gpu)
