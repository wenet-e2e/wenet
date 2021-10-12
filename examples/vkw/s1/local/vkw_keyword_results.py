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
from wenet.utils.common import get_subsample
from wenet.utils.common import remove_duplicates_and_blank
from wenet.utils.mask import make_pad_mask
from pypinyin import lazy_pinyin, Style

def map_words2pinyin(word_list_file):
    word_unit_dict = {}
    word_id_dict = {}    
    for line in open(word_list_file, mode="r", encoding="utf8"):
        ids, keyword = line.split("\n")[0].split()
        #keyword = line[0] 
        keyword_pinyin = lazy_pinyin(keyword, style=Style.TONE3, neutral_tone_with_five=True)
        keyword_pinyin = [j.replace('5','0') for j in keyword_pinyin]
        word_unit_dict[keyword] = keyword_pinyin
        word_id_dict[keyword] = ids
    return word_id_dict, word_unit_dict

def map_words2char(word_list_file):
    word_unit_dict = {}
    word_id_dict = {}    
    for line in open(word_list_file, mode="r", encoding="utf8"):
        ids, keyword = line.split("\n")[0].split()
        #keyword = line[0] 
        keyword_char = [i for i in keyword]
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
    #print(alignment)
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1

        if end == len(alignment) and start < end:
            if start == 0:
                timestamp.append(alignment[start:])
            else:
                timestamp[-1] += alignment[start:]

            #print(start, end, len(alignment), alignment)
            break

        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1

        timestamp.append(alignment[start:end])
        start = end
    #print(timestamp)
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

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description='use ctc to generate alignment')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--input_file', required=True, help='format data file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--keyword_unit_dict', required=True, help='dict file')
    parser.add_argument('--model_unit', required=True, 
                        choices=['char','yinjie'], 
                        default='char', help='dict file')
    parser.add_argument('--keyword_results', required=True, help='keyword results')
    parser.add_argument('--ctc_results', required=True, help='ctc results')
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    #print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    #if args.batch_size > 1:
    #    logging.fatal('alignment mode must be running with batch_size == 1')
    #    sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    ali_collate_conf = copy.deepcopy(configs['collate_conf'])
    ali_collate_conf['spec_aug'] = False
    ali_collate_conf['spec_sub'] = False
    ali_collate_conf['feature_dither'] = False
    ali_collate_conf['speed_perturb'] = False
    if raw_wav:
        ali_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    ali_collate_func = CollateFunc(**ali_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    ali_dataset = AudioDataset(args.input_file,
                               **dataset_conf,
                               raw_wav=raw_wav)
    ali_data_loader = DataLoader(ali_dataset,
                                 collate_fn=ali_collate_func,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=0)

    print("Reading: ", args.keyword_unit_dict)
    if args.model_unit == 'char':
        word_id_dict, word_unit_dict = map_words2char(args.keyword_unit_dict)
    else:
        word_id_dict, word_unit_dict = map_words2pinyin(args.keyword_unit_dict)
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
                for w in hyps[index]:
                    if w == eos:
                        break
                    content.append(char_dict[w])
                f_ctc_results.write('{} {}\n'.format(i, " ".join(content)))
            f_ctc_results.flush()

            for index, i in enumerate(key):
                timestamp = get_frames_timestamp(alignment[index])
                subsample = get_subsample(configs)
                word_seq, word_time = get_labformat_frames(timestamp, subsample, char_dict)
                for index_j in range(len(word_seq)):
                    for keyword in word_unit_list:
                        keyword_len = len(word_unit_dict[keyword])
                        if index_j+keyword_len > len(word_seq):
                            continue
                        if word_seq[index_j:index_j+keyword_len] == word_unit_dict[keyword]:
                            #print(word_id_dict[keyword], i, word_seq, word_time, index_j, index_j+keyword_len-1)
                            #print("{} {} {} {} {}".format(word_id_dict[keyword], i, word_time[index_j][0], word_time[index_j+keyword_len-1][1], 1.0))
                            f_keyword_results.write("{} {} {} {} {}\n".format(word_id_dict[keyword], i, word_time[index_j][0], word_time[index_j+keyword_len-1][1], 0.0))
            f_keyword_results.flush()        
    f_keyword_results.close()
    f_ctc_results.close()

if __name__ == "__main__":
    main()
