#!/usr/bin/env python3
# encoding: utf-8

import sys
import wave
import torchaudio

scp = sys.argv[1]
dur_scp = sys.argv[2]
sound_type = sys.argv[3]

with open(scp, 'r') as f, open(dur_scp, 'w') as fout:
    cnt = 0
    total_duration = 0
    for l in f:
        items = l.strip().split()
        wav_id = items[0]
        fname = items[1]
        cnt += 1
        frames = 0
        rate = 0
        if sound_type == "wav":
            with wave.open(fname, 'r') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
        elif sound_type == "flac":
            waveform, rate = torchaudio.load_wav(fname)
            frames = len(waveform[0])
        duration = frames / float(rate)
        total_duration += duration
        fout.write('{} {}\n'.format(wav_id, duration))
    print('process {} utts'.format(cnt))
    print('total {} s'.format(total_duration))
