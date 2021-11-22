# Pretrained Models in WeNet

## Model Types
We provide two types of pretrained model in WeNet to facilitate users with different requirements.

1. **Checkpoint Model**, with suffix **.pt**, the model trained and saved as checkpoint by WeNet python code, you can reproduce our published result with it, or you can use it as checkpoint to continue.

2. **Runtime Model**, with suffix **.zip**, you can directly use `runtime model` in our [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) or [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet) runtime, the `runtime model` is export by Pytorch JIT on the `checkpoint model`. Two kinds of runtime models are provided:
    * x86, server model, typically big.
    * android, on-device model, typically small and been quantized.

## Model License

The pretrained model in WeNet follows the license of it's corresponding dataset.
For example, the pretrained model on LibriSpeech follows `CC BY 4.0`, since it is used as license of the LibriSpeech dataset, see http://openslr.org/12/.

## Model List

Here is a list of the pretrained models on different datasets. The model structure, model size, and download link are given.


| Datasets                                    | Languages | Checkpoint Model                                                                                                                     | Runtime Model(x86)                                                                                                                          | Runtime Model(android)                                                                                                                     |
|---------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [aishell](../examples/aishell/s0/README.md) | CN        | Conformer/174M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20211025_conformer_exp.tar.gz) | U2 Transformer/127M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210601_unified_transformer_server.tar.gz) | U2 Transformer/38M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210601_unified_transformer_device.tar.gz) |
| [aishell2](../examples/aishell2/s0/README.md) | CN        | U2++ Conformer/187M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz) | U2 Transformer/130M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210602_unified_transformer_server.tar.gz) | U2 Transformer/39M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210602_unified_transformer_device.tar.gz) |
| [gigaspeech](../examples/gigaspeech/s0/README.md) | EN        | U2++ Conformer/472M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210811_conformer_bidecoder_exp.tar.gz) | U2++ Conformer/507M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210728_u2pp_conformer_server.tar.gz) | U2++ Transformer/51M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210823_u2pp_transformer_device.tar.gz) |
| [librispeech](../examples/librispeech/s0/README.md) | EN        | Conformer/481M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/librispeech/20210610_conformer_bidecoder_exp.tar.gz) | U2++ Conformer/199M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/librispeech/20210610_u2pp_conformer_server.tar.gz) |  |
| [multi_cn](../examples/multi_cn/s0/README.md) | CN        | U2 Conformer/193M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/multi_cn/20210815_unified_conformer_exp.tar.gz) | U2 Conformer/130M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/multi_cn/20210815_unified_conformer_server.tar.gz) | U2 Conformer/65M/[Download](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/multi_cn/20210815_unified_conformer_device.tar.gz) |
