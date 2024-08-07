# Python Package


## Install

``` sh
pip install git+https://github.com/wenet-e2e/wenet.git
```

## Development Install

``` sh
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .
```


## Command line Usage

``` sh
wenet --language chinese audio.wav
```

You can specify the following parameters.

* `-l` or `--language`: chinese/english are supported now.
* `-m` or `--model_dir`: your own model dir
* `-g` or `--gpu`: the device id of gpu, default value -1 represents for cpu.
* `-t` or `--show_tokens_info`: show the token level information such as timestamp, confidence, etc.
* `--align`: force align the input audio and transcript
* `--label`: the input label to align
* `--paraformer`: use the best Chinese model
* `--device`: specify the backend accelerator (cuda/npu/cpu)

## Python Programming Usage

``` python
import wenet

model = wenet.load_model('chinese')
# or model = wenet.load_model(model_dir='xxx')
result = model.transcribe('audio.wav')
print(result['text'])
```
