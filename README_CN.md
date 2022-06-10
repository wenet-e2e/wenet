# WeNet

[**English version**](https://github.com/wenet-e2e/wenet/tree/main/README.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**文档**](https://wenet-e2e.github.io/wenet/)
| [**训练模型教程 1**](https://wenet.org.cn/wenet/tutorial_librispeech.html)
| [**训练模型教程 2**](https://wenet.org.cn/wenet/tutorial_aishell.html)
| [**WeNet 论文**](https://wenet-e2e.github.io/wenet/papers.html)
| [**x86 识别服务**](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86)
| [**android 本地识别**](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet)



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
wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_libtorch.tar.gz
tar -xf 20210618_u2pp_conformer_libtorch.tar.gz
model_dir=$PWD/20210618_u2pp_conformer_libtorch
docker run --rm -it -p 10086:10086 -v $model_dir:/home/wenet/model wenetorg/wenet-mini:latest bash /home/run.sh
```

**实时识别**

使用浏览器打开文件`wenet/runtime/server/x86/web/templates/index.html`，在 `WebSocket URL` 中填入 `ws://127.0.0.1:10086` (若在windows下通过wsl2运行docker,  则使用`ws://localhost:10086`) , 允许浏览器弹出的请求使用麦克风，即可通过麦克风进行实时语音识别。

![Runtime web](/docs/images/runtime_web.png)


## 训练语音识别模型

**配置环境**

``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- 安装 Conda:  https://docs.conda.io/en/latest/miniconda.html
- 创建 Conda 环境:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

**训练模型**

使用中文 Aishell-1 数据集训练模型
```
cd examples/aishell/s0/
bash run.sh --stage -1
```

细节请阅读 [**训练模型教程**](https://wenet-e2e.github.io/wenet/tutorial_aishell.html)


### 新手常见问题

1. 请使用具有gpu的机器。确保cuda和torch都已经安装。wenet也支持cpu训练，但是速度非常很慢。
2. 请使用支持bash的环境。windows的默认cmd是不支持bash的。
3. run.sh脚本里，`export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`, 改为自己要用的GPU id，比如你的机器有4张GPU卡，4张卡都用来训练，改为 `export CUDA_VISIBLE_DEVICES="0,1,2,3"`
4. run.sh脚本里，`data=/export/data/asr-data/OpenSLR/33/`设置为自己的目录。请使用绝对路径而不要用相对路径。
5. 如果继续训练出错，请先删除实验目录下的 ddp_init文件再试一试。


## 技术支持

欢迎在 [Github Issues](https://github.com/wenet-e2e/wenet/issues) 中提交问题。

欢迎扫二维码加入微信讨论群，如果群人数较多，请添加右侧个人微信入群。

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

## 贡献者列表

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ---- | ---- | ---- | ---- | ---- |
| <a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="250px"></a> | <a href="https://thuhcsi.github.io" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/thu.png" width="250px"></a> | <a href="https://www.nvidia.com/en-us" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/nvidia.png" width="250px"></a> | | | |

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
  address={Brno, Czech Republic },
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
