#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader
torchaudio.set_audio_backend("sox")


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = torchaudio.backend.sox_backend.info(wav_path)[0].rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wa.scp
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.backend.sox_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    offset=start_frame)
                waveform = waveform * (1 << 15)
            else:
                waveform, sample_rate = torchaudio.load_wav(item[1])
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.feat_dim,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['collate_conf']['feature_extraction_conf']['mel_bins']

    collate_func = CollateFunc(feat_dim)
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
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size
            if wav_number % 1000 == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
