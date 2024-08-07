# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.audiollm_dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--output_len',
                        type=int,
                        default=100,
                        help='output length')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch size')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--temperature',
                        default=0.95,
                        type=float,
                        help='temperature')
    parser.add_argument('--top_p',
                        default=1.0,
                        type=float,
                        help='top_p')
    parser.add_argument('--top_k',
                        default=100,
                        type=int,
                        help='top_k')
    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False,
                           train=False)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    dir_name = os.path.join(args.result_dir, 
                            f"temp{args.temperature}_topk{args.top_k}_topp{args.top_p}")
    os.makedirs(dir_name, exist_ok=True)
    file_name = os.path.join(dir_name, 'text')
    file = open(file_name, 'w')

    stop_tokens = tokenizer.tokens2ids(["<|eot_id|>", "<|end_of_text|>"])

    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys = batch["keys"]
                results = model.generate(
                        batch,
                        device,
                        stop_tokens,
                        dtype,
                        args.output_len,
                        args.temperature,
                        args.top_p,
                        args.top_k)
                for i, (key, tokens) in enumerate(zip(keys, results)):
                    line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0])
                    logging.info('{}'.format(line))
                    file.write(line + '\n')
            file.close()
            
if __name__ == '__main__':
    main()
