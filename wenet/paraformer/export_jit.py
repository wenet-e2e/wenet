""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import argparse
import torch
import yaml
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import (SanmDecoer, SanmEncoder)
from wenet.paraformer.paraformer import Paraformer
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.cmvn import load_cmvn


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--ali_paraformer',
                        required=True,
                        help='ali released Paraformer model path')
    parser.add_argument('--config', required=True, help='config of paraformer')
    parser.add_argument('--output_file', default=None, help='output file')
    args = parser.parse_args()
    return args


def init_model(configs):
    mean, istd = load_cmvn(configs['cmvn_file'], True)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float())
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    encoder = SanmEncoder(global_cmvn=global_cmvn,
                          input_size=configs['lfr_conf']['lfr_m'] * input_dim,
                          **configs['encoder_conf'])
    decoder = decoder = SanmDecoer(vocab_size=vocab_size,
                                   encoder_output_size=encoder.output_size(),
                                   **configs['decoder_conf'])
    predictor = Cif(**configs['cif_predictor_conf'])
    model = Paraformer(
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
    )
    return model


def main():

    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(configs)
    load_checkpoint(model, args.ali_paraformer)
    model.eval()

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)


if __name__ == "__main__":

    main()
