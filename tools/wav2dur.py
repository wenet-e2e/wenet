import sys
import wave
import os
#scp = './wav.scp'
scp = sys.argv[1]
dur_scp = sys.argv[2]
with open(dur_scp, 'w') as fout:
    with open(scp,'r') as f:
        cnt = 0
        for l in f:
            items = l.strip().split(' ')
            wav_id = items[0]
            fname = items[1]
            cnt += 1
            with wave.open(fname,'r') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                fout.write('{} {}\n'.format(wav_id, duration))
        print('process {} utts'.format(cnt))
