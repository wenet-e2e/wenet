#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os


def sph2pipe_wav(in_wav, tmp_out_wav, out_wav):
    with open(in_wav, 'r', encoding='utf-8') as in_f:
        with open(tmp_out_wav, 'w', encoding='utf-8') as tmp_out_f:
            with open(out_wav, 'w', encoding='utf-8') as out_f:
                for line in in_f:
                    _tmp = line.strip().split(' ')
                    wav_out_path = _tmp[4]
                    wav_out_path = wav_out_path.split('/')
                    wav_out_path[-4] = wav_out_path[-4] + '_pipe'
                    if not os.path.exists('/'.join(wav_out_path[:-1])):
                        os.makedirs('/'.join(wav_out_path[:-1]))
                    wav_out_path = '/'.join(wav_out_path)
                    tmp_out_f.write(' '.join(_tmp[1:5]) + ' ' + wav_out_path +
                                    '\n')
                    out_f.write(_tmp[0] + ' ' + wav_out_path + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('wrong input parameter')
        raise NotImplementedError(len(sys.argv))
    in_wav = sys.argv[1]
    tmp_out_wav = sys.argv[2]
    out_wav = sys.argv[3]
    sph2pipe_wav(in_wav, tmp_out_wav, out_wav)
