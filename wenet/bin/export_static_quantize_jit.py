# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com), Lucky Wong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config


def get_args():
    parser = argparse.ArgumentParser(description='Static quantize your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--script_model',
                        required=True,
                        help='output script model')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str("-1")

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = 1
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    # Init asr model from configs
    model_fp32 = init_asr_model(configs)
    load_checkpoint(model_fp32, args.checkpoint)
    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    print('================ Float 32 ======================')
    print(model_fp32)

    # calibrate the prepared model to determine quantization parameters for
    # activations in a real world setting, the calibration would be done with
    # a representative dataset
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            model_fp32(feats, feats_lengths, target, target_lengths)
            if batch_idx % 100 == 0:
                print('Progress utts {}'.format(batch_idx))
                sys.stdout.flush()

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with
    # quantized implementations.
    print('=================== int8  ======================')
    model_int8 = torch.quantization.convert(model_fp32)
    print(model_int8)

    print('================ int8(script) ==================')
    script_model = torch.jit.script(model_int8)
    script_model.save(args.script_model)
    print(script_model)


if __name__ == '__main__':
    main()
