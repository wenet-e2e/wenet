""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import argparse
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from wenet.cif.predictor import Predictor
from wenet.paraformer.ali_paraformer.model import (
    AliParaformer,
    SanmDecoer,
    SanmEncoder,
)
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.cmvn import load_cmvn
from wenet.utils.file_utils import read_symbol_table


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--ali_paraformer',
                        required=True,
                        help='ali released Paraformer model path')
    parser.add_argument('--config', required=True, help='config of paraformer')
    parser.add_argument('--cmvn',
                        required=True,
                        help='cmvn file of paraformer in wenet style')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--wav', required=True, help='wav file')
    parser.add_argument('--output_file', default=None, help='output file')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    symbol_table = read_symbol_table(args.dict)
    char_dict = {v: k for k, v in symbol_table.items()}
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    mean, istd = load_cmvn(args.cmvn, is_json=True)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float())
    configs['encoder_conf']['input_size'] = 80 * 7
    encoder = SanmEncoder(global_cmvn=global_cmvn, **configs['encoder_conf'])
    configs['decoder_conf']['vocab_size'] = len(char_dict)
    configs['decoder_conf']['encoder_output_size'] = encoder.output_size()
    decoder = SanmDecoer(**configs['decoder_conf'])

    # predictor = PredictorV3(**configs['predictor_conf'])
    predictor = Predictor(**configs['predictor_conf'])
    model = AliParaformer(encoder, decoder, predictor)
    load_checkpoint(model, args.ali_paraformer)
    model.eval()

    waveform, sample_rate = torchaudio.load(args.wav)
    assert sample_rate == 16000
    waveform = waveform * (1 << 15)
    waveform = waveform.to(torch.float)
    feats = kaldi.fbank(waveform,
                        num_mel_bins=80,
                        frame_length=25,
                        frame_shift=10,
                        energy_floor=0.0,
                        sample_frequency=sample_rate)
    feats = feats.unsqueeze(0)
    feats_lens = torch.tensor([feats.size(1)], dtype=torch.int64)

    out, token_nums = model(feats, feats_lens)
    print("".join([char_dict[id] for id in out.argmax(-1)[0].numpy()]))
    print(token_nums)

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)

    model = torch.jit.load(args.output_file)
    out, token_nums = model.forward(feats, feats_lens)
    print("".join([char_dict[id] for id in out.argmax(-1)[0].numpy()]))
    print(token_nums)


if __name__ == "__main__":

    main()
