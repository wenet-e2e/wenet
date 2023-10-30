# Python Package


## Install

``` sh
pip install git+https://github.com/wenet-e2e/wenet.git
```

## Command line Usage

``` sh
wenet --language chinese audio.wav
```

You can specify the following parameters.

* `-l` or `--language`: chinese/english are supported now.
* `-m` or `--model_dir`: your own model dir
* `-t` or `--show_tokens_info`: show the token level information such as timestamp, confidence, etc.


## Python Programming Usage

``` python
import wenet

model = wenet.load_model('chinese')
# or model = wenet.load_model(model_dir='xxx')
result = model.transcribe('audio.wav')
print(result['text'])
```
