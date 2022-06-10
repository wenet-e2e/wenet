# WeNet Python Binding

This is a python binding of WeNet.

WeNet is a production first and production ready end-to-end speech recognition toolkit.

The best things of the binding are:

1. Multiple languages supports, including English, Chinese. Other languages are in development.
2. Non-streaming and streaming API
3. N-best, contextual biasing, and timestamp supports, which are very important for speech productions.
4. Alignment support. You can get phone level alignments this tool, on developing.

## Install

Python 3.6+ is required.

``` sh
pip3 install wenet
```

## Usage

### Non-streaming Usage

``` python
import sys
import wenet

model_dir = sys.argv[1]
wav_file = sys.argv[2]
decoder = wenet.Decoder(model_dir)
ans = decoder.decode_wav(wav_file)
print(ans)
# call decoder.reset() if you want to do the next decoding
```

`model_dir` is the `Runtime Model` directory of WeNet, it contains:
* `final.zip`: runtime TorchScript ASR model.
* `units.txt`: modeling units file
* `TLG.fst`: optional, it means decoding with LM when `TLG.fst` is given.
* `words.txt`: optional, word level symbol table for decoding with `TLG.fst`

Please refer https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md for the details of `Runtime Model`.

You can also specify the following parameter in `wenet.Decoder`

* `lang` (str): The language you used, `chs` for Chinese, and `en` for English.
* `nbest` (int): Output the top-n best result.
* `enable_timestamp` (bool): Whether to enable the word level timestamp.
* `context` (List[str]): a list of context biasing words.
* `context_score` (float): context bonus score.

For example:
``` python
decoder = wenet.Decoder(model_dir,
                        lang='chs',
                        nbest=5,
                        enable_timestamp=True,
                        context=['不忘初心', '牢记使命'],
                        context_score=3.0)
```

### Streaming Usage

``` python
import sys
import wave
import wenet

model_dir = sys.argv[1]
test_wav = sys.argv[2]

with wave.open(test_wav, 'rb') as fin:
    assert fin.getnchannels() == 1
    wav = fin.readframes(fin.getnframes())

decoder = wenet.Decoder(model_dir)
# We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
interval = int(0.5 * 16000) * 2
for i in range(0, len(wav), interval):
    last = False if i + interval < len(wav) else True
    chunk_wav = wav[i: min(i + interval, len(wav))]
    ans = decoder.decode(chunk_wav, last)
    print(ans)
```

You can use the same parameters as we introduced above to control the behavior of `wenet.Decoder`


## Build on Your Local Machine

``` sh
git clone git@github.com:wenet-e2e/wenet.git
cd wenet/runtime/binding/python
python setup.py install
```

