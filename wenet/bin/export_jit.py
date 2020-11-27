# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

from __future__ import print_function

import argparse
import os

import yaml
import torch

from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.ctc import CTC
from wenet.transformer.asr_model import ASRModel
from wenet.utils.checkpoint import load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', required=True, help='output file')

    args = parser.parse_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin)

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim, **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim, **configs['encoder_conf'])
    decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                 **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())
    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model
    script_model = torch.jit.script(model)
    script_model.save(args.output_file)

    print('Succeed, see {}'.format(args.output_file))
