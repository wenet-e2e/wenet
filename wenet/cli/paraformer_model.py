import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.cli.hub import Hub
from wenet.paraformer.search import paraformer_beautify_result, paraformer_greedy_search
from wenet.utils.file_utils import read_symbol_table


class Paraformer:

    def __init__(self, model_dir: str, device: int = -1) -> None:

        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        if device >= 0:
            device = 'cuda:{}'.format(device)
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        symbol_table = read_symbol_table(units_path)
        self.char_dict = {v: k for k, v in symbol_table.items()}
        self.eos = 2

    def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float).to(self.device)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=16000)
        feats = feats.unsqueeze(0)
        feats_lens = torch.tensor([
            feats.size(1)],
            dtype=torch.int64,
            device=feats.device)

        decoder_out, token_num = self.model.forward_paraformer(
            feats, feats_lens)

        res = paraformer_greedy_search(decoder_out, token_num)[0]

        result = {}
        result['confidence'] = res.confidence
        # # TODO(Mddct): deal with '@@' and 'eos'
        result['text'] = paraformer_beautify_result(
            [self.char_dict[x] for x in res.tokens])

        if tokens_info:
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.char_dict[x],
                    # TODO(Mddct): support times
                    # 'start': 0,
                    # 'end': 0,
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info

        # result = ''.join(hyp)
        return result

    def align(self, audio_file: str, label: str) -> dict:
        raise NotImplementedError("Align is currently not supported")


def load_model(model_dir: str = None, gpu: int = -1) -> Paraformer:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang('paraformer')
    return Paraformer(model_dir, gpu)
