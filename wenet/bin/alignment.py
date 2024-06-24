# Copyright (c) 2021 Mobvoi Inc. (authors: Di Wu)
#               2022 Tinnove Inc (authors: Wei Ren)
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
from textgrid import TextGrid, IntervalTier
import math

from wenet.dataset.dataset import Dataset
from wenet.utils.ctc_utils import force_align
from wenet.utils.common import get_subsample
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer


def generator_textgrid(maxtime, lines, output):
    # Download Praat: https://www.fon.hum.uva.nl/praat/
    interval = maxtime / (len(lines) + 1)
    margin = 0.0001

    tg = TextGrid(maxTime=maxtime)
    linetier = IntervalTier(name="line", maxTime=maxtime)

    i = 0
    for l in lines:
        s, e, w = l.split()
        linetier.add(minTime=float(s) + margin, maxTime=float(e), mark=w)

    tg.append(linetier)
    print("successfully generator {}".format(output))
    tg.write(output)


def get_frames_timestamp(alignment,
                         prob,
                         blank_thres=0.999,
                         thres=0.0000000001):
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    timestamp = []
    # get frames level duration for each token
    start = 0
    end = 0
    local_start = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1
        if end == len(alignment):
            timestamp[-1] += alignment[start:]
            break
        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1
        local_start = end - 1
        # find the possible front border for current token
        while local_start >= start and (
                prob[local_start][0] < math.log(blank_thres)
                or prob[local_start][alignment[end - 1]] > math.log(thres)):
            alignment[local_start] = alignment[end - 1]
            local_start -= 1
        cur_alignment = alignment[start:end]
        timestamp.append(cur_alignment)
        start = end
    return timestamp


def get_labformat(timestamp, subsample):
    begin = 0
    begin_time = 0
    duration = 0
    labformat = []
    for idx, t in enumerate(timestamp):
        # 25ms frame_length,10ms hop_length, 1/subsample
        subsample = get_subsample(configs)
        # time duration
        i = 0
        while t[i] == 0:
            i += 1
        begin = i
        dur = 0
        while i < len(t) and t[i] != 0:
            i += 1
            dur += 1
        begin = begin_time + begin * 0.01 * subsample
        duration = dur * 0.01 * subsample
        if idx < len(timestamp) - 1:
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            char_dict[t[-1]]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, char_dict[t[-1]]))
        else:  # last token
            non_blank = 0
            for i in t:
                if i != 0:
                    token = i
                    break
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            char_dict[token]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, char_dict[token]))
        begin_time += len(t) * 0.01 * subsample
    return labformat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use ctc to generate alignment')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--input_file', required=True, help='format data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--blank_thres',
                        default=0.999999,
                        type=float,
                        help='ctc blank thes')
    parser.add_argument('--thres',
                        default=0.000001,
                        type=float,
                        help='ctc non blank thes')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument(
        '--non_lang_syms',
        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--result_file',
                        required=True,
                        help='alignment result file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gen_praat',
                        action='store_true',
                        help='convert alignment to a praat format')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.batch_size > 1:
        logging.fatal('alignment mode must be running with batch_size == 1')
        sys.exit(1)

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

    # Init dataset and data loader
    ali_conf = copy.deepcopy(configs['dataset_conf'])

    ali_conf['filter_conf']['max_length'] = 102400
    ali_conf['filter_conf']['min_length'] = 0
    ali_conf['filter_conf']['token_max_length'] = 102400
    ali_conf['filter_conf']['token_min_length'] = 0
    ali_conf['filter_conf']['max_output_input_ratio'] = 102400
    ali_conf['filter_conf']['min_output_input_ratio'] = 0
    ali_conf['speed_perturb'] = False
    ali_conf['spec_aug'] = False
    ali_conf['spec_trim'] = False
    ali_conf['shuffle'] = False
    ali_conf['sort'] = False
    ali_conf['fbank_conf']['dither'] = 0.0
    ali_conf['batch_conf']['batch_type'] = "static"
    ali_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    ali_dataset = Dataset(args.data_type,
                          args.input_file,
                          tokenizer,
                          ali_conf,
                          partition=False)

    ali_data_loader = DataLoader(ali_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model, configs = init_model(args, configs)

    device = torch.device(args.device)
    model = model.to(device)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w',
                               encoding='utf-8') as fout:
        for batch_idx, batch in enumerate(ali_data_loader):
            print("#" * 80)
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
            # print(ctc_probs.size(1))
            ctc_probs = ctc_probs.squeeze(0)
            target = target.squeeze(0)
            alignment = force_align(ctc_probs, target)
            fout.write('{} {}\n'.format(key[0], alignment))

            if args.gen_praat:
                timestamp = get_frames_timestamp(alignment, ctc_probs,
                                                 args.blank_thres, args.thres)
                subsample = get_subsample(configs)
                labformat = get_labformat(timestamp, subsample)

                lab_path = os.path.join(os.path.dirname(args.result_file),
                                        key[0] + ".lab")
                with open(lab_path, 'w', encoding='utf-8') as f:
                    f.writelines(labformat)

                textgrid_path = os.path.join(os.path.dirname(args.result_file),
                                             key[0] + ".TextGrid")
                generator_textgrid(maxtime=(len(alignment) + 1) * 0.01 *
                                   subsample,
                                   lines=labformat,
                                   output=textgrid_path)
