# WeNet Roadmap

This roadmap for WeNet.
WeNet is a community-driven project and we love your feedback and proposals on where we should be heading.

Please open up [issues](https://github.com/wenet-e2e/wenet/issues/) or
[discussion](https://github.com/wenet-e2e/wenet/discussions) on github to write your proposal.
Feel free to volunteer yourself if you are interested in trying out some items(they do not have to be on the list).


## WeNet 3.0 (2023.06)

- [x] ONNX support, see https://github.com/wenet-e2e/wenet/pull/1103
- [x] RNN-T support, see https://github.com/wenet-e2e/wenet/pull/1261
- [ ] Self training, streaming
- [ ] Light weight, low latency, on-device model exploration
  - [x] TrimTail, see https://github.com/wenet-e2e/wenet/pull/1487/, [paper link](https://arxiv.org/pdf/2211.00522.pdf)
- [ ] Audio-Visual speech recognition
- [ ] OS or Hardware Platforms
  - [x] IOS, https://github.com/wenet-e2e/wenet/pull/1549
  - [x] Raspberry Pi, see https://github.com/wenet-e2e/wenet/pull/1477
  - [ ] Harmony OS
- [ ] ASIC XPU
  - [x] Horizon X3 pi, BPU, see https://github.com/wenet-e2e/wenet/pull/1597
  - [x] Kunlun XPU, see https://github.com/wenet-e2e/wenet/pull/1455
- [x] Public Model Hub Support
  - [x] HuggingFace, see https://huggingface.co/spaces/wenet/wenet_demo
  - [x] ModelScope, see https://modelscope.cn/models/wenet/u2pp_conformer-asr-cn-16k-online/summary
 - [x] [Vosk](https://github.com/alphacep/vosk-api/) like models and API for developers.
    - [x] Models(Chinese/English/Japanese/Korean/French/German/Spanish/Portuguese)
      - [x] Chinese
      - [x] English
    - [x] API(python/c/c++/go/java)
      - [x] python

## WeNet 2.0 (2022.06)

- [x] U2++ framework for better accuracy
- [x] n-gram + WFST language model solution
- [x] Context biasing(hotword) solution
- [x] Very big data training support with UIO
- [x] More dataset support, including WenetSpeech, GigaSpeech, HKUST and so on.

## WeNet 1.0 (2021.02)

- [x] Streaming solution(U2 framework)
- [x] Production runtime solution with `TorchScript` training and `LibTorch` inference.
- [x] Unified streaming and non-streaming model(U2 framework)

