# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#                    Tencent (Yougen Yuan)
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

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

from wenet.utils.common import get_subsample
from wenet.utils.common import remove_duplicates_and_blank
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.mask import make_pad_mask


def map_words2char(word_list_file):
    word_unit_dict = {}
    word_id_dict = {}
    for line in open(word_list_file, mode="r", encoding="utf8"):
        ids, keyword = line.split("\n")[0].split()
        keyword_char = []
        for i in keyword:
            keyword_char.append(i)
        word_unit_dict[keyword] = keyword_char
        word_id_dict[keyword] = ids
    return word_id_dict, word_unit_dict


def get_frames_timestamp(alignment):
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    timestamp = []
    # get frames level duration for each token
    start = 0
    end = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1

        if end == len(alignment) and start < end:
            if start == 0:
                timestamp.append(alignment[start:])
            else:
                timestamp[-1] += alignment[start:]
            break

        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1

        timestamp.append(alignment[start:end])
        start = end
    return timestamp


def get_labformat_frames(timestamp, subsample, char_dict):
    begin = 0
    duration = 0
    word_seq = []
    word_time = []
    for idx, t in enumerate(timestamp):
        duration = len(t) * subsample
        if idx < len(timestamp) - 1:
            word_seq.append(char_dict[t[-1]])
            word_time.append([begin, begin + duration])
        else:
            non_blank = 0
            token = 0
            for i in t:
                if i != 0:
                    token = i
                    break
            word_seq.append(char_dict[token])
            word_time.append([begin, begin + duration])
        begin = begin + duration
    return word_seq, word_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--input_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
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
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument('--keyword_unit_dict',
                        required=True,
                        help='keyword id')
    parser.add_argument('--keyword_results',
                        required=True,
                        help='keyword results')
    parser.add_argument('--ctc_results', required=True, help='ctc results')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    distributed = args.world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)

    symbol_table = read_symbol_table(args.symbol_table)
    # Load dict
    char_dict = {}
    with open(args.symbol_table, mode='r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False

    cv_dataset = Dataset(args.data_type,
                         args.input_data,
                         symbol_table,
                         cv_conf,
                         None,
                         partition=False)

    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    print("Reading: ", args.keyword_unit_dict)
    word_id_dict, word_unit_dict = map_words2char(args.keyword_unit_dict)
    word_unit_list = list(word_unit_dict.keys())
    print("word_unit_list has the size of %d" % (len(word_unit_list)))

    # Init asr model from configs
    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    f_keyword_results = open(args.keyword_results, 'w', encoding='utf-8')
    f_ctc_results = open(args.ctc_results, 'w', encoding='utf-8')
    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_data_loader):
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
            batch_size = encoder_out.size(0)
            ctc_probs = model.ctc.log_softmax(
                encoder_out)  # (1, maxlen, vocab_size)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
            topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
            mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
            topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)
            alignment = [hyp.tolist() for hyp in topk_index]
            hyps = [remove_duplicates_and_blank(hyp) for hyp in alignment]
            for index, i in enumerate(key):
                content = []
                if len(hyps[index]) > 0:
                    for w in hyps[index]:
                        if w == eos:
                            break
                        content.append(char_dict[w])
                f_ctc_results.write('{} {}\n'.format(i, " ".join(content)))
            f_ctc_results.flush()
            for index, i in enumerate(key):
                timestamp = get_frames_timestamp(alignment[index])
                subsample = get_subsample(configs)
                word_seq, word_time = get_labformat_frames(
                    timestamp, subsample, char_dict)
                for index_j in range(len(word_seq)):
                    for keyword in word_unit_list:
                        keyword_len = len(word_unit_dict[keyword])
                        if index_j + keyword_len > len(word_seq):
                            continue
                        if (word_seq[index_j:index_j +
                                     keyword_len] == word_unit_dict[keyword]):
                            f_keyword_results.write("{} {} {} {} {}\n".format(
                                word_id_dict[keyword], i,
                                word_time[index_j][0],
                                word_time[index_j + keyword_len - 1][1], 0.0))
            f_keyword_results.flush()
    f_keyword_results.close()
    f_ctc_results.close()
