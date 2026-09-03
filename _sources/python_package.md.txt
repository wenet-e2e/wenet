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

## GPU/Hardware Notes

Measured on a Tesla T4 (16GB, Turing/sm_75), torch 2.4.0+cu121.

* **GPU is not the default.** `--device` (CLI) and `device` (`wenet.load_model`)
  both default to `'cpu'`. On a GPU host, `wenet -m <model> audio.wav` still
  runs on CPU unless you pass `--device cuda`. There is no warning when this
  happens. Measured on paraformer: CUDA is **2.84x** faster than CPU
  (0.205s vs 0.581s per utterance, cached model, T4).
* **No dtype flag on the pip-installed CLI.** The `wenet` console command
  (`wenet/cli/transcribe.py`) always runs fp32 and has no `--dtype`/`--fp16`
  flag. The training-recipe script `wenet/bin/recognize.py` does expose
  `--dtype {fp16,fp32,bf16}`, but that is a separate, more involved entry
  point (local recipe checkout, not the pip package).
* **bf16 can be slower than fp32 on Turing-class GPUs (e.g. T4).** T4 lacks
  bf16 tensor-core acceleration. On WeNet's own Conformer models (e.g.
  `wenetspeech`), casting to bf16 measured **~47% slower** than fp32
  (46.6ms vs 31.7ms per forward pass). fp16 does not crash but shows no
  speedup at batch size 1 with short audio (launch-overhead bound), though it
  does cut VRAM by about 46%.
* **`use_sdpa` is off by default in the shipped configs, and turning it on
  helps most on bf16.** WeNet's own Conformer implementation supports a
  `use_sdpa: true` toggle (`encoder_conf`/`decoder_conf`) that switches
  attention from a manual matmul+softmax to
  `torch.nn.functional.scaled_dot_product_attention`, using the same
  checkpoint with no retraining. The `train.yaml` shipped with pretrained
  models (e.g. `wenetspeech`) does not set this key, so it loads as `False`.
  Enabling it measured 7-29% faster on a T4, with identical output text:
  fp32 31.7ms -> 29.4ms (~7%), fp16 32.2ms -> 30.0ms (~7%), bf16
  46.6ms -> 33.1ms (~29%, cutting most of the bf16 slowdown above). This
  toggle only applies to WeNet's own Conformer models; Paraformer (SANM
  attention) does not have a `use_sdpa` code path.
* **int8 dynamic quantization is CPU-only.** `wenet/bin/export_jit.py` uses
  `torch.quantization.quantize_dynamic`, which works on CPU but raises
  `NotImplementedError: Could not run 'quantized::linear_dynamic' with
  arguments from the 'CUDA' backend` if applied to a CUDA model — this is a
  PyTorch backend limitation, not a WeNet bug. (`wenet/bin/export_onnx_cpu.py`
  quantizes separately via `onnxruntime.quantization.quantize_dynamic`, a
  different, ONNX-graph-level API that isn't affected by this PyTorch
  limitation.) For GPU-side reduced precision, see
  `wenet/bin/export_onnx_gpu.py --fp16` (ONNX export, a separate fp16-only
  workflow).
