import sys
import wave

import wenet


if len(sys.argv) != 3:
    print('Usage: {} model_dir test.wav'.format(sys.argv[0]))
    sys.exit(1)

model_dir = sys.argv[1]
test_wav = sys.argv[2]

with wave.open(test_wav, 'rb') as fin:
    assert fin.getnchannels() == 1
    wav = fin.readframes(fin.getnframes())

# Optional set log level
# wenet.set_log_level(2)

# Init decoder
decoder = wenet.Decoder(model_dir)
# decoder.set_nbest(10)
# decoder.enable_timestamp(True)
# decoder.add_context(['停滞'])
# decoder.set_context_score(3.0)

# Non-streaming decode
print('-----Non-streaming decoding demo-----')
ans = decoder.decode(wav)
print(ans)
decoder.reset()  # reset status after we finish decoding

# Stream decode
# We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
interval = int(0.5 * 16000) * 2
print('-----Streaming decoding demo-----')
for i in range(0, len(wav), interval):
    last = False if i + interval < len(wav) else True
    chunk_wav = wav[i: min(i + interval, len(wav))]
    ans = decoder.decode(chunk_wav, last)
    print(ans)
