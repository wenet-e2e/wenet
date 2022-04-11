# wenet-python
python wrapper for wenet runtime
- non streamming example
```python

model = wenet.load_model("../model")
model.recognize("test.wav")
#[{"sentence":"甚至出现交易几乎停滞的情况","word_pieces":[{"word":"甚","start":0,"end":840},{"word":"至","start":840,"end":1080},{"word":"出","start":1080,"end":1320},{"word":"现","start":1320,"end":1640},{"word":"交","start":1640,"end":1880},{"word":"易","start":1880,"end":2120},{"word":"几","start":2120,"end":2320},{"word":"乎","start":2320,"end":2560},{"word":"停","start":2560,"end":2800},{"word":"滞","start":2800,"end":2960},{"word":"的","start":2960,"end":3200},{"word":"情","start":3200,"end":3560},{"word":"况","start":3560,"end":4120}]}]
```
- streamming example
```python streamming decoding

import wenet
import time

params = wenet.Params()
params.chunk_size = 16
params.model_path = "../model/final.zip"
params.dict_path = "../model/words.txt"

model = wenet.WenetWrapper(params)
with open("../model/test.wav", "rb") as f:
    b = f.read()
b = b[44:]

decoder = wenet.StreammingAsrDecoder(model)
def send():
    nbytes = int((16000/1000)*2*800)
    final = False
    r = nbytes
    l = 0
    while True:
        if r == len(b):
            final = True

        yield b[l:r] , final

        l = r
        if l >= len(b):
            break

        if r + nbytes < len(b):
            r = r + nbytes
        else:
            r = len(b)

res = decoder.streamming_recognize(send())
for r in res:
    print(r)
```
```bash
[{"sentence":"甚至"}]
[{"sentence":"甚至出现"}]
[{"sentence":"甚至出现交易几"}]
[{"sentence":"甚至出现交易几乎停滞"}]
[{"sentence":"甚至出现交易几乎停滞的情"}]
[{"sentence":"甚至出现交易几乎停滞的情况","word_pieces":[{"word":"甚","start":0,"end":880},{"word":"至","start":880,"end":1120},{"word":"出","start":1120,"end":1400},{"word":"现","start":1400,"end":1720},{"word":"交","start":1720,"end":1960},{"word":"易","start":1960,"end":2120},{"word":"几","start":2120,"end":2400},{"word":"乎","start":2400,"end":2640},{"word":"停","start":2640,"end":2800},{"word":"滞","start":2800,"end":3040},{"word":"的","start":3040,"end":3240},{"word":"情","start":3240,"end":3600},{"word":"况","start":3600,"end":4120}]}]
[{"sentence":"甚至"}]
[{"sentence":"甚至出现"}]
[{"sentence":"甚至出现交易几"}]
[{"sentence":"甚至出现交易几乎停滞"}]
[{"sentence":"甚至出现交易几乎停滞的情"}]
[{"sentence":"甚至出现交易几乎停滞的情况","word_pieces":[{"word":"甚","start":0,"end":880},{"word":"至","start":880,"end":1120},{"word":"出","start":1120,"end":1400},{"word":"现","start":1400,"end":1720},{"word":"交","start":1720,"end":1960},{"word":"易","start":1960,"end":2120},{"word":"几","start":2120,"end":2400},{"word":"乎","start":2400,"end":2640},{"word":"停","start":2640,"end":2800},{"word":"滞","start":2800,"end":3040},{"word":"的","start":3040,"end":3240},{"word":"情","start":3240,"end":3600},{"word":"况","start":3600,"end":4120}]}]
```
- label check
```python
import wenet

model = wenet.load_model("../model")

labels =  ["甚", "至", "出", "现", "交", "易", "几", "乎", "停", "滞", "的", "情", "好"]
checker = wenet.LabelChecker(model)
checker.check("../model/test.wav", labels)
```
```bash
[{"sentence":"甚至出现交易几乎停滞的情<is>况</is>","word_pieces":[{"word":"甚","start":0,"end":840},{"word":"至","start":840,"end":1080},{"word":"出","start":1080,"end":1320},{"word":"现","start":1320,"end":1640},{"word":"交","start":1640,"end":1880},{"word":"易","start":1880,"end":2120},{"word":"几","start":2120,"end":2320},{"word":"乎","start":2320,"end":2560},{"word":"停","start":2560,"end":2800},{"word":"滞","start":2800,"end":2960},{"word":"的","start":2960,"end":3200},{"word":"情","start":3200,"end":3560},{"word":"况","start":3560,"end":4120}]}]
```
