# WeNet

[**中文版**](https://github.com/wenet-e2e/wenet/blob/main/README_CN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Google Groups**](https://groups.google.com/g/wenet-e2e)
| [**Docs**](https://wenet-e2e.github.io/wenet/)
| [**Tutorial**](https://wenet-e2e.github.io/wenet/tutorial.html)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime (x86)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86)
| [**Runtime (android)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet)

**We** share neural **Net** together.

The main motivation of WeNet is to close the gap between research and production end-to-end (E2E) speech recognition models,
to reduce the effort of productionizing E2E models, and to explore better E2E models for production.

## Highlights

* **Production first and production ready**: The python code of WeNet meets the requirements of TorchScript,
  so the model trained by WeNet can be directly exported by Torch JIT and use LibTorch for inference.
  There is no gap between the research model and production model.
  Neither model conversion nor additional code is required for model inference.
* **Unified solution for streaming and non-streaming ASR**: WeNet implements [Unified Two Pass (U2)](https://arxiv.org/pdf/2012.05481.pdf)
  framework to achieve accurate, fast and unified E2E model, which is favorable for industry adoption.
* **Portable runtime**: Several demos will be provided to show how to host WeNet trained models
  on different platforms, including server [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) and on-device [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet).
* **Light weight**: WeNet is designed specifically for E2E speech recognition,
  with clean and simple code. It is all based on PyTorch and its corresponding ecosystem. It has no dependency on Kaldi,
  which simplifies installation and usage.

## Performance Benchmark

Please see `examples/$dataset/s0/README.md` for benchmark on different speech datasets.
* [AIShell-1](examples/aishell/s0/README.md)
* [AIShell-2](examples/aishell2/s0/README.md)
* [LibriSpeech](examples/librispeech/s0/README.md)
* [GigaSpeech](examples/gigaspeech/s0/README.md)
* [Multi-CN](examples/multi_cn/s0/README.md) trained using all open source Chinese corpus.

## Installation

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env: (**PyTorch 1.6.0** is recommended. We met some error on NCCL when using 1.7.0 on 2080 Ti.)

``` sh
# [option 1]
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch==1.6.0 cudatoolkit=10.1 torchaudio=0.6.0 -c pytorch

# [option 2: working on machine with GPU 3090]
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch torchvision torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/server/x86
mkdir build && cd build && cmake .. && cmake --build .
```

## Discussion & Communication

Please join our [Google Groups](https://groups.google.com/g/wenet-e2e) for further discussion.

For Chinese users, in addition to Google Groups, You can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

If you can not access the QR image, please access it on [gitee](https://gitee.com/robin1001/qr/tree/master).

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

## Contributors

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ---- | ---- | ---- | ---- | ---- |

## Acknowledge

1. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet) for transformer based modeling.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for WFST based decoding for LM integration.
3. We referred [EESEN](https://github.com/srvk/eesen) for building TLG based graph for LM integration.
4. We referred to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/) for python batch inference of e2e models.

## Citations

``` bibtex
@inproceedings{yao2021wenet,
  title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
  author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
  booktitle={Proc. Interspeech},
  year={2021},
  address={Brno, Czech Republic }
  organization={IEEE}
}

@article{zhang2020unified,
  title={Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition},
  author={Zhang, Binbin and Wu, Di and Yao, Zhuoyuan and Wang, Xiong and Yu, Fan and Yang, Chao and Guo, Liyong and Hu, Yaguang and Xie, Lei and Lei, Xin},
  journal={arXiv preprint arXiv:2012.05481},
  year={2020}
}

@article{wu2021u2++,
  title={U2++: Unified Two-pass Bidirectional End-to-end Model for Speech Recognition},
  author={Wu, Di and Zhang, Binbin and Yang, Chao and Peng, Zhendong and Xia, Wenjing and Chen, Xiaoyu and Lei, Xin},
  journal={arXiv preprint arXiv:2106.05642},
  year={2021}
}
```
