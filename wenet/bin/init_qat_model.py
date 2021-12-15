# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
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
from wenet.utils.checkpoint import save_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config


def load_quant_checkpoint(model: torch.nn.Module, path: str) -> dict:
    def insert_scope(state_name, flag='linear'):
        state_name_split = state_name.split('.')
        state_name_split.insert(-1, flag)
        return '.'.join(state_name_split)

    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        state_dict = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        state_dict = torch.load(path, map_location='cpu')

    # FIXME(Lucky): use module type to mapping state dictionary (nn.Linear, nn.Conv1d, nn.Conv2d).
    quant_state_dict = {}
    for k,v in state_dict.items():
        if k.startswith('encoder.embed.conv'):
            quant_state_dict[insert_scope(k, 'qconv2d')] = v
        elif k.startswith('encoder.embed.out'):
            quant_state_dict[insert_scope(k, 'qlinear')] = v
        elif k.endswith('.depthwise_conv.weight') or k.endswith('.depthwise_conv.bias'):
            quant_state_dict[insert_scope(k, 'qconv1d')] = v
        elif k.find('pointwise_conv') >= 0:
            quant_state_dict[insert_scope(k, 'qconv1d')] = v
        elif k.find('norm') >= 0 or k.find('embed') >= 0:
            quant_state_dict[k] = v
        elif k.endswith('.weight') or k.endswith('.bias'):
            quant_state_dict[insert_scope(k, 'qlinear')] = v
        else:
            quant_state_dict[k] = v

    for k,v in model.state_dict().items():
        if k not in quant_state_dict:
            if k.find('fake_quant')>0 or k.find('max_val')>0 or k.find('min_val')>0 or k.find('post_process')>0:
                pass
            else:
                print(k)

    model.load_state_dict(quant_state_dict, strict=False)

    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def get_args():
    parser = argparse.ArgumentParser(description='Static quantize your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output', required=True, help='Quantized checkpoint model')
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

    # Init asr model from configs
    model_fp32 = init_asr_model(configs)
    load_quant_checkpoint(model_fp32, args.checkpoint)
    save_checkpoint(model_fp32, args.output, infos=None)

if __name__ == '__main__':
    main()
