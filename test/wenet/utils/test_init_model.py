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
        if config.get('cmvn', None) == "global_cmvn":
            config['cmvn_conf']['cmvn_file'] = "test/resources/global_cmvn"
        if 'tokenizer' in config:
            if config['tokenizer'] == "char":
                config['tokenizer_conf'][
                    'symbol_table_path'] = "test/resources/aishell2.words.txt"
            elif config['tokenizer'] == "bpe":
                config['tokenizer_conf']['bpe_path'] = \
                    "test/resources/librispeech.train_960_unigram5000.bpemodel"
                config['tokenizer_conf']['symbol_table_path'] = \
                    "test/resources/librispeech.words.txt"
                config['tokenizer_conf']['non_lang_syms_path'] = \
                    "test/resources/non-linguistic-symbols.invalid"
            elif config['tokenizer'] == "whisper":
                config['tokenizer_conf']['is_multilingual'] = True
                config['tokenizer_conf']['num_languages'] = 100
            elif config['tokenizer'] == 'paraformer':
                config['tokenizer_conf'][
                    'symbol_table_path'] = "test/resources/paraformer.words.txt"
                config['tokenizer_conf'][
                    'seg_dict_path'] = "test/resources/paraformer.seg_dict.txt"
            else:
                raise NotImplementedError
        else:
            config['tokenizer'] = "char"
            config['tokenizer_conf'] = {}
            config['tokenizer_conf']['symbol_table_path'] = \
                "test/resources/aishell2.words.txt"
            config['tokenizer_conf']['non_lang_syms_path'] = \
                "test/resources/non-linguistic-symbols.invalid"
        print("checking {} {}".format(c, config))
        init_model(args, config)
