# WeNet Python Binding

## Usage

``` python
import sys
import wave

import wenet

model_dir = sys.argv[1]
test_wav = sys.argv[2]

with wave.open(test_wav, 'rb') as fin:
   assert fin.getnchannels() == 1
   wav = fin.readframes(fin.getnframes())

# Init wenet decoder
decoder = wenet.Decoder(model_dir)
# Decode and get result
ans = decoder.decode(wav)
print(ans)
```
