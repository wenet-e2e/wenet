import librosa
# import os
import sys

def mincut(wavscpfn, minsec):
    outfn = wavscpfn + "_" + str(minsec)

    with open(outfn, 'w') as bw:
        with open(wavscpfn) as br:
            for aline in br.readlines():
                aline = aline.strip()
                afn = aline.split('\t')[1]
                # print(afn)
                dur = librosa.get_duration(filename=afn)
                if dur >= minsec:
                    bw.write(aline + '\n')

# wn.3.mincut.py <wav.scp> <min.sec>
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('{} <in.wav.scp> <min.sec.cut>'.format(sys.argv[0]))
        exit()

    wavscpfn = sys.argv[1]
    minsec = float(sys.argv[2])

    mincut(wavscpfn, minsec)
