# WeNet

[**中文版**](https://github.com/wenet-e2e/wenet/blob/main/README_CN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](ROADMAP.md)
| [**Docs**](https://wenet-e2e.github.io/wenet/)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime (x86)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86)
| [**Runtime (android)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet)
| [**Pretrained Models**](docs/pretrained_models.md)

**We** share neural **Net** together.

The main motivation of WeNet is to close the gap between research and production end-to-end (E2E) speech recognition models,
to reduce the effort of productionizing E2E models, and to explore better E2E models for production.

## Highlights

* **Production first and production ready**: The core design principle of WeNet. WeNet provides full stack solutions for speech recognition.
  * *Unified solution for streaming and non-streaming ASR*: [U2 framework](https://arxiv.org/pdf/2012.05481.pdf)--develop, train, and deploy only once.
  * *Runtime solution*: built-in server [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) and on-device [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet) runtime solution.
  * *Model exporting solution*: built-in solution to export model to LibTorch/ONNX for inference.
  * *LM solution*: built-in production-level [LM solution](docs/lm.md).
  * *Other production solutions*: built-in contextual biasing, time stamp, endpoint, and n-best solutions.

* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.

## Performance Benchmark

Please see `examples/$dataset/s0/README.md` for benchmark on different speech datasets.

## Installation(Python Only)

If you just want to use WeNet as a python package for speech recognition application,
just install it by `pip`, please note python 3.6+ is required.
``` sh
pip3 install wenet
```

And please see [doc](runtime/binding/python/README.md) for usage.


## Installation(Training and Developing)

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/server/x86
mkdir build && cd build && cmake .. && cmake --build .
```

## Discussion & Communication

Please visit [Discussions](https://github.com/wenet-e2e/wenet/discussions) for further discussion.

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

If you can not access the QR image, please access it on [gitee](https://gitee.com/robin1001/qr/tree/master).

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

## Contributors

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ---- | ---- | ---- | ---- | ---- |
| <a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="250px"></a> | <a href="https://thuhcsi.github.io" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/thu.png" width="250px"></a> | <a href="https://www.nvidia.com/en-us" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/nvidia.png" width="250px"></a> | | | |

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
  address={Brno, Czech Republic },
  organization={IEEE}
}

@article{zhang2022wenet,
  title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
  author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}
```
