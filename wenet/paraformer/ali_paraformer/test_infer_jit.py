""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import argparse
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.init_model import init_model


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
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    model = init_model(configs)
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

    decode_results = model.decode(['paraformer_greedy_search'], feats,
                                  feats_lens)
    print("".join([
        char_dict[id]
        for id in decode_results['paraformer_greedy_search'][0].tokens
    ]))

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)

    model = torch.jit.load(args.output_file)
    out, token_nums = model.forward(feats, feats_lens)
    print("".join([char_dict[id] for id in out.argmax(-1)[0].numpy()]))
    print(token_nums)


if __name__ == "__main__":

    main()
