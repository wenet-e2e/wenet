#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
import json

def kaldi2json(kaldi_cmvn_file):
    means = []
    variance = []
    with open(kaldi_cmvn_file, 'r') as fid:
        # kaldi binary file start with '\0B'
        if fid.read(2) == '\0B':
            logging.error('kaldi cmvn binary file is not supported, please '
                          'recompute it by: compute-cmvn-stats --binary=false '
                          ' scp:feats.scp global_cmvn')
            sys.exit(1)
        fid.seek(0)
        arr = fid.read().split()
        assert (arr[0] == '[')
        assert (arr[-2] == '0')
        assert (arr[-1] == ']')
        feat_dim = int((len(arr) - 2 - 2) / 2)
        for i in range(1, feat_dim + 1):
            means.append(float(arr[i]))
        count = float(arr[feat_dim + 1])
        for i in range(feat_dim + 2, 2 * feat_dim + 2):
            variance.append(float(arr[i]))

    cmvn_info = {'mean_stat:' : means,
                 'var_stat' : variance,
                 'frame_num' : count}
    return cmvn_info

if __name__ == '__main__':
    with open(sys.argv[2], 'w') as fout:
        cmvn = kaldi2json(sys.argv[1])
        fout.write(json.dumps(cmvn))
