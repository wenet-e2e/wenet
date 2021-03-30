# Copyright (c) 2021 Mobvoi Inc. (authors: Di Wu)
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

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.ctc_util import forced_align

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use ctc to generate alignment')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--input_file', required=True, help='data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.batch_size > 1:
        logging.fatal(
            'alignment mode must be running with batch_size == 1')
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    ali_collate_conf = copy.deepcopy(configs['collate_conf'])
    ali_collate_conf['spec_aug'] = False
    ali_collate_conf['spec_sub'] = False
    ali_collate_conf['feature_dither'] = False
    ali_collate_conf['speed_perturb'] = False
    if raw_wav:
        ali_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    ali_collate_func = CollateFunc(**ali_collate_conf,
                                   raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    ali_dataset = AudioDataset(args.input_file, **dataset_conf, raw_wav=raw_wav)
    ali_data_loader = DataLoader(ali_dataset,
                                 collate_fn=ali_collate_func,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w', encoding='utf-8') as fout:
        for batch_idx, batch in enumerate(ali_data_loader):
            key, feat, target, feats_length, target_length = batch
            feat = feat.to(device)
            target = target.to(device)
            feats_length = feats_length.to(device)
            target_length = target_length.to(device)
            # Let's assume B = batch_size and N = beam_size
            # 1. Encoder
            encoder_out, encoder_mask = model._forward_encoder(
                feat, feats_length)  # (B, maxlen, encoder_dim)
            maxlen = encoder_out.size(1)
            ctc_probs = model.ctc.log_softmax(
                encoder_out)  # (1, maxlen, vocab_size)
            ctc_probs = ctc_probs.squeeze(0)
            target = target.squeeze(0)
            alignment = forced_align(ctc_probs, target)
            fout.write('{} {}\n'.format(key[0], alignment))
