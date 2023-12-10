#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-10] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import glob
import yaml

from wenet.utils.init_model import init_model


class DummyArguments:
    jit = False
    enc_init = None
    checkpoint = None


def test_init_model():
    configs = glob.glob("examples/*/*/conf/*.yaml")
    args = DummyArguments()
    for c in configs:
        with open(c, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        if 'fbank_conf' in config['dataset_conf']:
            input_dim = config['dataset_conf']['fbank_conf']['num_mel_bins']
        elif 'log_mel_spectrogram_conf' in config['dataset_conf']:
            input_dim = config['dataset_conf']['log_mel_spectrogram_conf'][
                'num_mel_bins']
        else:
            input_dim = config['dataset_conf']['mfcc_conf']['num_mel_bins']
        config['input_dim'] = input_dim
        # TODO(xcsong): fix vocab_size
        config['output_dim'] = 3000
        print("checking {} {}".format(c, config))
        init_model(args, config)
