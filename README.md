# WeNet

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](https://github.com/wenet-e2e/wenet/issues/1683)
| [**Docs**](https://wenet-e2e.github.io/wenet)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime (x86)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/libtorch)
| [**Runtime (android)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/android)
| [**Pretrained Models**](docs/pretrained_models.md)
| [**HuggingFace**](https://huggingface.co/spaces/wenet/wenet_demo)

**We** share neural **Net** together.

The main motivation of WeNet is to close the gap between research and production end-to-end (E2E) speech recognition models,
to reduce the effort of productionizing E2E models, and to explore better E2E models for production.

## :fire: News

* 2022.12: Horizon X3 pi BPU, see https://github.com/wenet-e2e/wenet/pull/1597, Kunlun Core XPU, see https://github.com/wenet-e2e/wenet/pull/1455, Raspberry Pi, see https://github.com/wenet-e2e/wenet/pull/1477, IOS, see https://github.com/wenet-e2e/wenet/pull/1549.
* 2022.11: TrimTail paper released, see https://arxiv.org/pdf/2211.00522.pdf
* 2022.10: Squeezeformer is supported, see https://github.com/wenet-e2e/wenet/pull/1447.
* 2022.07: RNN-T is supported now, see [rnnt](https://github.com/wenet-e2e/wenet/tree/main/examples/aishell/rnnt) for benchmark.

## Highlights

* **Production first and production ready**: The core design principle of WeNet. WeNet provides full stack solutions for speech recognition.
  * *Unified solution for streaming and non-streaming ASR*: [U2++ framework](https://arxiv.org/pdf/2203.15455.pdf)--develop, train, and deploy only once.
  * *Runtime solution*: built-in server [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/libtorch) and on-device [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/android) runtime solution.
  * *Model exporting solution*: built-in solution to export model to LibTorch/ONNX for inference.
  * *LM solution*: built-in production-level [LM solution](docs/lm.md).
  * *Other production solutions*: built-in contextual biasing, time stamp, endpoint, and n-best solutions.

* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.

## Performance Benchmark

Please see `examples/$dataset/s0/README.md` for benchmark on different speech datasets.

## Install
please refer [doc](docs/install.md) for install.


## Discussion & Communication

Please visit [Discussions](https://github.com/wenet-e2e/wenet/discussions) for further discussion.

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

If you can not access the QR image, please access it on [gitee](https://gitee.com/robin1001/qr/tree/master).

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

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
