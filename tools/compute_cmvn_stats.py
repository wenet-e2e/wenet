import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import sys
import json

MEL_BINS=80
number = 0
mean_stat = np.zeros(MEL_BINS)
var_stat = np.zeros(MEL_BINS)
wav_number=0
with open(sys.argv[1],'r') as fin:
    for l in fin:
        items = l.strip().split(' ')
        waveform, sample_rate = torchaudio.load_wav(items[1])
        mat = kaldi.fbank(
                        waveform,
                        num_mel_bins=MEL_BINS,
                        dither=0.0,
                        energy_floor=0.0
                    )
        mat = mat.detach().numpy()
        mean_stat +=np.sum(mat, axis=0)
        var_stat +=np.sum(np.square(mat), axis=0)
        number +=mat.shape[0]
        wav_number +=1
        if wav_number % 1000 ==0:
            print('process {} wavs'.format(wav_number))

cmvn_info = {'mean_stat:':list(mean_stat), 
         'var_stat':list(var_stat),
         'frame_num':number}

with open(sys.argv[2],'w') as fout:
    fout.write(json.dumps(cmvn_info))