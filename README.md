# WeNet

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](https://github.com/wenet-e2e/wenet/issues/1683)
| [**Docs**](https://wenet-e2e.github.io/wenet)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime**](https://github.com/wenet-e2e/wenet/tree/main/runtime)
| [**Pretrained Models**](docs/pretrained_models.md)
| [**HuggingFace**](https://huggingface.co/spaces/wenet/wenet_demo)

**We** share **Net** together.

## News :fire:

* 2022.12: Horizon X3 pi BPU, see https://github.com/wenet-e2e/wenet/pull/1597, Kunlun Core XPU, see https://github.com/wenet-e2e/wenet/pull/1455, Raspberry Pi, see https://github.com/wenet-e2e/wenet/pull/1477, IOS, see https://github.com/wenet-e2e/wenet/pull/1549.
* 2022.11: TrimTail paper released, see https://arxiv.org/pdf/2211.00522.pdf

## Highlights

* **Production first and production ready**: The core design principle, WeNet provides full stack production solutions for speech recognition.
* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.


## Install
please refer [doc](docs/install.md) for install.


## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |


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
