# WeNet

We share net together.
We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet),
and we refered to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/blob/master/otrans/recognizer.py)
for batch inference.

The main motivation of WeNet is to close the gap between research and production End-to-End (E2E) speech recognition models,
to reduce the effort of productizing E2E models, and to explore better E2E models for production.

## Highlights

* **Light weight**: WeNet is designed specifically for E2E speech recognition,
  with clean and simple code. It is all based on PyTorch and its corresponding ecosystem.
* **Production ready**: The python code of WeNet meets the requirements of TorchScript,
  so the model trained by WeNet can be directly exported by Torch JIT and use LibTorch for inference.
  There is no gap between the research model and production model.
  Neither model conversion nor additional code is required for model inference.
* **Portable runtime**: WeNet will provide several demos to show how to host WeNet trained models
  on different platforms, including server (x86) and embedded (ARM in Android platforms).
* **Unified solution for streaming and non-streaming ASR**: WeNet implements [Unified Two Pass (U2)](https://arxiv.org/pdf/2012.05481.pdf)
  framework to achieve accurate, fast and unified E2E model, which is favorable for industry adoption.
* **Well documented**: Detailed documentation and tutorials will be provided.

## Performance Benchmark

Please see examples/$dataset/s0/README.md for WeNet benchmark on different speech datasets.
* [AIShell-1](examples/aishell/s0/README.md)
* LibriSpeech (to be added)

## Documentation

You can visit [Docs](https://mobvoi.github.io/wenet/) for WeNet Sphinx documentation. Or please see tutorails below:
* [Tutorial](docs/tutorial.md)
* [JIT in WeNet](docs/jit_in_wenet.md)
* [Runtime](docs/runtime.md)

## Installation

- Clone
``` sh
git clone https://github.com/mobvoi/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create conda env: **pytorch 1.6.0** is suggested. We meet some error on NCCL when using 1.7.0 on 2080 Ti.

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

- Install Kaldi: WeNet requries Kaldi for feature extraction (a TorchAudio version is under development).
  Please download and build [Kaldi](https://github.com/kaldi-asr/kaldi), then set Kaldi root as:

``` sh
vim example/aishell/s0/path.sh
KALDI_ROOT=${your_kaldi_root_path}
```
