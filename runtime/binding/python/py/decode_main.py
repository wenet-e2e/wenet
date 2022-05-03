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
wenet.set_log_level(1)

# Init decoder
decoder = wenet.Decoder(model_dir)
ans = decoder.decode(wav)
print('Decode the first utterance, result: {}'.format(ans))

# If you want decode anther wav, just reset decoder, and decode
for i in range(10):
    decoder.reset()
    ans = decoder.decode(wav)
    print('Reset and decode utterance {}, result: {}'.format(i, ans))
