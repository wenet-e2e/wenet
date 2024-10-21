# WeNet

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](https://github.com/wenet-e2e/wenet/issues/1683)
| [**Docs**](https://wenet-e2e.github.io/wenet)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime**](https://github.com/wenet-e2e/wenet/tree/main/runtime)
| [**Pretrained Models**](docs/pretrained_models.md)
| [**HuggingFace**](https://huggingface.co/spaces/wenet/wenet_demo)
| [**Ask WeNet Guru**](https://gurubase.io/g/wenet)

**We** share **Net** together.

## Highlights

* **Production first and production ready**: The core design principle, WeNet provides full stack production solutions for speech recognition.
* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.


## Install

### Install python package

``` sh
pip install git+https://github.com/wenet-e2e/wenet.git
```

**Command-line usage** (use `-h` for parameters):

``` sh
wenet --language chinese audio.wav
```

**Python programming usage**:

``` python
import wenet

model = wenet.load_model('chinese')
result = model.transcribe('audio.wav')
print(result['text'])
```

Please refer [python usage](docs/python_package.md) for more command line and python programming usage.

### Install for training & deployment

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.10
conda activate wenet
conda install conda-forge::sox
```

- Install CUDA: please follow this [link](https://icefall.readthedocs.io/en/latest/installation/index.html#id1), It's recommended to install CUDA 12.1
- Install torch and torchaudio, It's recomended to use 2.2.2+cu121:

``` sh
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

<details><summary><b>For Ascend NPU users:</b></summary>

- Install CANN: please follow this [link](https://ascend.github.io/docs/sources/ascend/quick_install.html) to install CANN toolkit and kernels.

- Install WeNet with torch-npu dependencies:

``` sh
pip install -e .[torch-npu]
```

- Related version control table:

| Requirement  |      Minimum     | Recommend   |
| ------------ | ---------------- | ----------- |
| CANN         | 8.0.RC2.alpha003 | latest      |
| torch        | 2.1.0            | 2.2.0       |
| torch-npu    | 2.1.0            | 2.2.0       |
| torchaudio   | 2.1.0            | 2.2.0       |
| deepspeed    | 0.13.2           | latest      |

</details>

- Install other python packages

``` sh
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
```

- Frequently Asked Questions (FAQs)

``` sh
# If you encounter sox compatibility issues
RuntimeError: set_buffer_size requires sox extension which is not available.
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
# conda env
conda install  conda-forge::sox
```

**Build for deployment**

Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```

Please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime) for building
runtime on more platforms and OS.


## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

For Chinese users, you can also scan the QR code on the left to follow our official account of WeNet.
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
