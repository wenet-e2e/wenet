# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import argparse
import os

import torch
import yaml

import wenet.dataset.processor as processor
from wenet.cli.hub import Hub
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer


def load_or_download(model_name_or_path):
    if model_name_or_path in Hub.assets:
        model_dir = Hub.download_model(model_name_or_path)
    else:
        model_dir = model_name_or_path
    return model_dir


def load_tokenizer(model_name_or_path):
    model_dir = load_or_download(model_name_or_path)
    config_file = os.path.join(model_dir, 'train.yaml')
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    for key, value in configs['tokenizer_conf'].items():
        if isinstance(value, str):
            rewrite_path = os.path.join(model_dir, os.path.basename(value))
            if os.path.exists(rewrite_path):
                configs['tokenizer_conf'][key] = rewrite_path
    return init_tokenizer(configs)


def load_feature(model_name_or_path):
    model_dir = load_or_download(model_name_or_path)
    config_file = os.path.join(model_dir, 'train.yaml')
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    conf = configs['dataset_conf']
    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    feats_conf = conf.get(f'{feats_type}_conf', {})
    feats_func = getattr(processor, f'compute_{feats_type}')
    feature_dim = feats_conf.get('num_mel_bins', 80)

    def compute_feature(wav_file):
        sample = {'key': wav_file, 'wav': wav_file}
        sample = processor.decode_wav(sample)
        sample = processor.resample(sample, 16000)
        sample = feats_func(sample, **feats_conf)
        return sample['feat']

    return compute_feature, feature_dim


def load_model(model_name_or_path, device='cpu'):
    model_dir = load_or_download(model_name_or_path)
    """ There are the follow files in in `model_dir`
        * final.pt, required
        * train.yaml, required
        * units.txt, required
        * global_cmvn, optional
    """
    # Check required files
    required_files = ['train.yaml', 'final.pt', 'units.txt']
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file {file} not found in {model_dir}")
    # Read config and override some config
    config_file = os.path.join(model_dir, 'train.yaml')
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    cmvn_file = os.path.join(model_dir, 'global_cmvn')
    if os.path.exists(cmvn_file):
        configs['cmvn_conf']['cmvn_file'] = cmvn_file
    # Read model
    pt_file = os.path.join(model_dir, 'final.pt')
    args = argparse.Namespace()
    args.checkpoint = pt_file
    # load model
    model, configs = init_model(args, configs)
    # load and set tokenizer
    tokenizer = load_tokenizer(model_dir)
    setattr(model, 'tokenizer', tokenizer)  # noqa, dynamic inject
    # load and set feature function
    compute_feature, _ = load_feature(model_dir)
    setattr(model, 'compute_feature', compute_feature)  # noqa, dynamic inject

    if next(model.parameters()).device == torch.device('meta'):
        print('model is on a meta device, this is for huggingface transformer')
    else:
        model = model.to(device)
    return model
