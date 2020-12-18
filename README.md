# WeNet

We share net together.
We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet),
and we refered to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/blob/master/otrans/recognizer.py)
for batch inference.

The main motivation of WeNet is to close the gap between research and production End to End(E2E) speech recognition model,
to reduce the efforts of productizing E2E model, and to explore better E2E model for production.

## Installation

WeNet requires PyTorch 1.6.0.

``` sh
# 1. setup your own python3 virtual env, miniconda is recommended.
# 2. install pytorch: conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# 3. install requirements: pip install -r requirements.txt
# 4. link Kaldi on root directory of this repo: ln -s YOUR_KALDI_PATH kaldi
```

## Feature

* **Light weight**: Wenet is specifically designed for E2E speech recognition,
  the code of which is clean, simple, and all based on PyTorch and it's corresponding ecosystem.
* **Production prefered**: The python code of WeNet meets the requirements TorchScript,
  so the model trained by WeNet can be directly exported by torch JIT, and be inferenced by LibTorch.
  There is no gap in the research model and prodction model,
  neither model conversion or additional code is required for model inference.
* **Production runtime**: WeNet will provide several demos to show how to host WeNet trained models
  on different platforms, including x86, ARM and Android platforms.
* **Unified streaming and non-streaming solution**: WeNet implements [Unified Two Pass(U2)](https://arxiv.org/pdf/2012.05481.pdf)
  framework to give accurate, fast and unified E2E model, which is industry favored.
* **Well documented**

## Performance

Please see examples/$dataset/s0/README.md for WeNet benchmark on different speech datasets.
* [AIShell-1](examples/aishell/s0/README.md)

