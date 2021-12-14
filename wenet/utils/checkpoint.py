# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
import os
import re

import yaml
import torch


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = True) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint, strict=strict)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


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

def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)
