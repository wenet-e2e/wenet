# WeNet

[**English version**](https://github.com/wenet-e2e/wenet/tree/main/README.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/mobvoi/wenet)

[**文档**](https://wenet-e2e.github.io/wenet/)
| [**训练模型教程**](https://wenet-e2e.github.io/wenet/tutorial.html)
| [**WeNet 论文**](https://wenet-e2e.github.io/wenet/papers.html)
| [**x86 识别服务**](https://github.com/mobvoi/wenet/tree/main/runtime/server/x86)
| [**android 本地识别**](https://github.com/mobvoi/wenet/tree/main/runtime/device/android/wenet)



## 核心功能

WeNet 是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务，其主要特点如下：

* 使用 conformer 网络结构和 CTC/attention loss 联合优化方法，统一的流式/非流式语音识别方案，具有业界一流的识别效果。
* 提供云上和端上直接部署的方案，最小化模型训练和产品落地之间的工程工作。
* 框架简洁，模型训练部分完全基于 pytorch 生态，不依赖于 kaldi 等复杂的工具。
* 详细的注释和文档，非常适合用于学习端到端语音识别的基础知识和实现细节。
* 支持时间戳，对齐，端点检测，语言模型等相关功能。


## 1分钟 Demo

**使用预训练模型和 docker 进行语音识别，1分钟（如果网速够快）搭建一个语音识别系统**

下载官方提供的预训练模型，并启动 docker 服务，加载模型，提供 websocket 协议的语音识别服务。

``` sh
wget http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210602_unified_transformer_server.tar.gz
tar -xf 20210602_unified_transformer_server.tar.gz
model_dir=$PWD/20210602_unified_transformer_server
docker run --rm -it -p 10086:10086 -v $model_dir:/home/wenet/model mobvoiwenet/wenet:mini bash /home/run.sh
```

**实时识别**

使用浏览器打开文件`wenet/runtime/server/x86/web/templates/index.html`，在 `WebSocket URL` 中填入 `ws://127.0.0.1:10086`, 允许浏览器弹出的请求使用麦克风，即可通过麦克风进行实时语音识别。

![Runtime web](/docs/images/runtime_web.png)


## 训练语音识别模型

**配置环境**

``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- 安装 Conda:  https://docs.conda.io/en/latest/miniconda.html
- 创建 Conda 环境: (推荐**PyTorch 1.6.0**. 在2080 Ti上使用1.7.0会有NCCL的问题)

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch==1.6.0 cudatoolkit=10.1 torchaudio=0.6.0 -c pytorch

# GPU 3090
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch torchvision torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

**训练模型**

使用中文 Aishell-1 数据集训练模型
```
cd examples/aishell/s0/
bash run.sh --stage -1
```

细节请阅读 [**训练模型教程**](https://wenet-e2e.github.io/wenet/tutorial.html)


## WeNet 性能

WeNet 提供了一些开源数据集的脚本，具体的模型性能如下，注意**其中提供的预训练模型为 pytorch 训练时使用的模型，并非 runtime 模型**。runtime 模型需要进行导出操作。
* [AIShell-1](examples/aishell/s0/README.md) 中文模型。
* [AIShell-2](examples/aishell2/s0/README.md) 中文模型。
* [LibriSpeech](examples/librispeech/s0/README.md) 英文模型。
* [Multi-CN](examples/multi_cn/s0/README.md) 使用所有开源中文数据集训练的中文模型。


## 技术支持

欢迎在 [Github Issues](https://github.com/mobvoi/wenet/issues) 中提交问题。

欢迎扫二维码加入微信讨论群，如果群人数较多，请添加右侧个人微信入群。

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

## 贡献者列表

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ---- | ---- | ---- | ---- | ---- |

## 致谢

WeNet 借鉴了一些优秀的开源项目，包括

1. Transformer 建模 [ESPnet](https://github.com/espnet/espnet)
2. WFST 解码 [Kaldi](http://kaldi-asr.org/)
3. TLG 构图 [EESEN](https://github.com/srvk/eesen)
4. Python Batch 推理 [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/)

## 引用

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
