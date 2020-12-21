# WeNet

We share net together.
We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet),
and we refered to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/blob/master/otrans/recognizer.py)
for batch inference.

The main motivation of WeNet is to close the gap between research and production End to End(E2E) speech recognition model,
to reduce the efforts of productizing E2E model, and to explore better E2E model for production.

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

- Install Kaldi: WeNet requries Kaldi to extract feature (a torchaudio version is in developing),
  please download and build [Kaldi](https://github.com/kaldi-asr/kaldi), then set Kaldi root as:

``` sh
vim example/aishell/s0/path.sh
KALDI_ROOT=your_kaldi_root_path
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

