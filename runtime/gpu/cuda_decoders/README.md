## Using CUDA based Decoders for Triton ASR Server
### Introduction
The triton model repository `model_repo_cuda_decoder` here, integrates the [CUDA WFST decoder](https://github.com/nvidia-riva/riva-asrlib-decoder) originally described in https://arxiv.org/abs/1910.10032. We take small conformer fp16 onnx inference for offline ASR as an example.

### Quick Start
```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
docker pull soar97/triton-wenet:22.12
docker run -it --rm --name "wenet_tlg_test" --gpus all --shm-size 1g --net host soar97/triton-wenet:22.12
# inside the docker container
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/runtime/gpu/cuda_wfst_decoder
# Use pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple if you encounter network issue
pip3 install -r requirements.txt

# For offline TLG ASR
bash run.sh
# For streaming TLG ASR
bash run_streaming.sh
```

### Performance of Small Offline ASR Model using Different Decoders

Benchmark(small offline conformer onnx fp16 model trained on Aishell1) based on Aishell1 test set with V100, the total audio duration is 36108.919 seconds.

(Note: 80 concurrent tasks, service has been fully warm up.)
|Decoding Method | decoding time(s) | WER (%) |
|----------|--------------------|-------------|
| CTC Greedy Search             | 23s | 4.97  |
| CUDA TLG 1-best (3-gram LM)   | 31s | 4.58  |
