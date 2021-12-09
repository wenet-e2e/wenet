import sys
import json
import soundfile as sf
import numpy as np


def generate(wav_file, out):
    """
    inp: a single channel, 16kHz wav file
    out: the generated json test data
    """
    print("Reading {}".format(wav_file))
    waveform, sample_rate = sf.read(wav_file)
    batch_size = 1
    mat = np.array([waveform] * batch_size, dtype=np.float32)

    out_dict = {"data": [{"WAV_LENS": [len(waveform)],
                "WAV": {"shape": [len(waveform)], "content": mat.flatten().tolist()}}]}
    json.dump(out_dict, open(out, "w"))

if __name__ == "__main__":
    inp = sys.argv[1]
    out = sys.argv[2]
    generate(inp, out)
