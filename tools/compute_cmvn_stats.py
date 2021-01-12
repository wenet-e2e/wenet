#!/usr/bin/env python3
# encoding: utf-8

import argparse
import json
import codecs

from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

MEL_BINS = 80
class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''
    def __init__(self):
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(MEL_BINS)
        var_stat = torch.zeros(MEL_BINS)
        number = 0
        for item in batch:
            key = item[0]
            waveform, sample_rate = torchaudio.load_wav(item[1])
            mat = kaldi.fbank(waveform,
                              num_mel_bins=MEL_BINS,
                              dither=0.0,
                              energy_floor=0.0)
            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat

class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

parser = argparse.ArgumentParser(description='extract CMVN stats')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='num of subprocess workers for processing')
parser.add_argument('--in_scp', default=None, help='wav scp file')
parser.add_argument('--out_cmvn', default='global_cmvn', help='global cmvn file')

args = parser.parse_args()

collate_func = CollateFunc()
dataset = AudioDataset(args.in_scp)
batch_size = 20
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         sampler=None,
                         num_workers=args.num_workers,
                         collate_fn=collate_func)

with torch.no_grad():
    all_number = 0
    all_mean_stat = torch.zeros(MEL_BINS)
    all_var_stat = torch.zeros(MEL_BINS)
    wav_number = 0
    for i, batch in enumerate(data_loader):
        number, mean_stat, var_stat = batch
        all_mean_stat += mean_stat
        all_var_stat += var_stat
        all_number += number
        wav_number += batch_size
        if wav_number % 1000 == 0:
            print('process {} wavs,{} frames'.format(wav_number, all_number))

cmvn_info = {'mean_stat' : list(all_mean_stat.tolist()), 
             'var_stat' : list(all_var_stat.tolist()),
             'frame_num' : all_number}

with open(args.out_cmvn, 'w') as fout:
    fout.write(json.dumps(cmvn_info))
