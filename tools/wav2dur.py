#!/usr/bin/env python3
# encoding: utf-8

import sys

import torchaudio
torchaudio.set_audio_backend("sox")
from tools.process_pipe_input import _process_pipe_input

scp = sys.argv[1]
dur_scp = sys.argv[2]

with open(scp, 'r') as f, open(dur_scp, 'w') as fout:
    cnt = 0
    total_duration = 0
    for l in f:
        items = l.strip().split()
        wav_id = items[0]
        fname = l.strip().lstrip(wav_id+" ")
        cnt += 1
        if fname.endswith("|"): # if wav_path is a shell command
            waveform, rate = _process_pipe_input(fname)
        else:
            waveform, rate = torchaudio.load_wav(fname)
        frames = len(waveform[0])
        duration = frames / float(rate)
        total_duration += duration
        fout.write('{} {}\n'.format(wav_id, duration))
    print('process {} utts'.format(cnt))
    print('total {} s'.format(total_duration))
