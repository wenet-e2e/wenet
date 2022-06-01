import os
import json
import soundfile as sf
import numpy as np
import argparse
import math

def generate_offline_input(args):
    wav_file = args.audio_file
    print("Reading {}".format(wav_file))
    waveform, sample_rate = sf.read(wav_file)
    batch_size = 1
    mat = np.array([waveform] * batch_size, dtype=np.float32)

    out_dict = {"data": [{"WAV_LENS": [len(waveform)],
                "WAV": {"shape": [len(waveform)],
                        "content": mat.flatten().tolist()}}]}
    json.dump(out_dict, open("offline_input.json", "w"))

def generate_online_input(args):
    wav_file = args.audio_file
    waveform, sample_rate = sf.read(wav_file)
    chunk_size, subsampling = args.chunk_size, args.subsampling
    context = args.context
    first_chunk_length = (chunk_size - 1) * subsampling + context
    frame_length_ms, frame_shift_ms = args.frame_length_ms, args.frame_shift_ms
    # for the first chunk,
    # we need additional frame to generate the exact first chunk length frames
    add_frames = math.ceil((frame_length_ms - frame_shift_ms) / frame_shift_ms)
    first_chunk_ms = (first_chunk_length + add_frames) * frame_shift_ms
    other_chunk_ms = chunk_size * subsampling * frame_shift_ms
    first_chunk_s = first_chunk_ms / 1000
    other_chunk_s = other_chunk_ms / 1000

    wav_segs = []
    i = 0
    while i < len(waveform):
        if i == 0:
            stride = int(first_chunk_s * sample_rate)
            wav_segs.append(waveform[i: i + stride])
        else:
            stride = int(other_chunk_s * sample_rate)
            wav_segs.append(waveform[i: i + stride])
        i += len(wav_segs[-1])

    data = {"data": [[]]}

    for idx, seg in enumerate(wav_segs):  # 0, num_frames + 5, 64
        chunk_len = len(seg)
        if idx == 0:
            length = int(first_chunk_s * sample_rate)
            expect_input = np.zeros((1, length), dtype=np.float32)
        else:
            length = int(other_chunk_s * sample_rate)
            expect_input = np.zeros((1, length), dtype=np.float32)

        expect_input[0][0:chunk_len] = seg

        flat_chunk = expect_input.flatten().astype(np.float32).tolist()
        seq = {"WAV": {"content": flat_chunk, "shape": expect_input[0].shape},
               "WAV_LENS": [chunk_len]}
        data["data"][0].append(seq)

    json.dump(data, open("online_input.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file',
                        type=str, default=None,
                        help='single wav file')
    # below is only for streaming input
    parser.add_argument('--streaming',
                        action="store_true", required=False)
    parser.add_argument('--sample_rate',
                        type=int, required=False, default=16000,
                        help='sample rate used in training')
    parser.add_argument('--frame_length_ms',
                        type=int, required=False, default=25,
                        help='frame length used in training')
    parser.add_argument('--frame_shift_ms',
                        type=int, required=False, default=10,
                        help='frame shift length used in training')
    parser.add_argument('--chunk_size',
                        type=int, required=False, default=16,
                        help='chunk size default is 16')
    parser.add_argument('--context',
                        type=int, required=False, default=7,
                        help='conformer context default is 7')
    parser.add_argument('--subsampling',
                        type=int, required=False, default=4,
                        help='subsampling rate default is 4')

    args = parser.parse_args()

    if args.streaming and os.path.exists(args.audio_file):
        generate_online_input(args)
    else:
        generate_offline_input(args)
