# Pretrained Models in WeNet

## Model Types
We provide two types of pretrained model in WeNet to facilitate users with different requirements.

1. **Checkpoint Model**, with suffix **.pt**, the model trained and saved as checkpoint by WeNet python code, you can reproduce our published result with it, or you can use it as checkpoint to continue.

2. **Runtime Model**, with suffix **.zip**, you can directly use `runtime model` in our [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) or [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet) runtime, the `runtime model` is export by Pytorch JIT on the `checkpoint model`. And the runtime models has been quantized to reduce the model size and network traffic.

## Model License

The pretrained model in WeNet follows the license of it's corresponding dataset.
For example, the pretrained model on LibriSpeech follows `CC BY 4.0`, since it is used as license of the LibriSpeech dataset, see http://openslr.org/12/.

## Model List

Here is a list of the pretrained models on different datasets. The model structure, model size, and download link are given.

| Datasets  | Languages     | Checkpoint Model  | Runtime Model     | Contributor |
|---    |---    |---    |---    |---    |
| [aishell](../examples/aishell/s0/README.md)   | CN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_exp.tar.gz)  | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_libtorch.tar.gz)     | <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a> |
| [aishell2](../examples/aishell2/s0/README.md)     | CN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_exp.tar.gz)     | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_libtorch.tar.gz)    | <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a> |
| [gigaspeech](../examples/gigaspeech/s0/README.md)     | EN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_exp.tar.gz)   | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_libtorch.tar.gz)  |  <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a> |
| [librispeech](../examples/librispeech/s0/README.md)   | EN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_exp.tar.gz)  | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_libtorch.tar.gz)     |  <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="100px"></a> |
| [multi_cn](../examples/multi_cn/s0/README.md)     | CN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/multi_cn/20210815_unified_conformer_exp.tar.gz)  | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/multi_cn/20210815_unified_conformer_libtorch.tar.gz)     | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="100px"></a> |
| [wenetspeech](../examples/wenetspeech/s0/README.md)     | CN    | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20220506_u2pp_conformer_exp.tar.gz) | [Conformer](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20220506_u2pp_conformer_libtorch.tar.gz)     | <a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="100px"></a> |
