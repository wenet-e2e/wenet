### Using Tensorrt for Triton ASR Server

```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
docker pull soar97/triton-wenet:22.12
docker run -it --rm --name "wenet_trt_test" --gpus all --shm-size 1g --net host soar97/triton-wenet:22.12
# inside the docker container
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/runtime/gpu/tensorrt
pip3 install nvidia-pyindex
# Use pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple if you encounter network issue
pip3 install -r requirements.txt

bash run_streaming_small_model.sh
```

#### Performance of Small u2pp Model for Streaming ASR

Benchmark(small u2pp onnx) based on Aishell1 test set with server-A10 (16vCPU 60GB Memory)/client(4vCPU 16GB Memory), the total audio duration is 36108.919 seconds.

(Note: using non-simulate-streaming mode)
|concurrent-tasks | processing time(s) |
|----------|--------------------|
| 20 (onnx fp16)                | 123.796 |
| 40 (onnx fp16)                | 84.557  |
| 60 (onnx fp16)                | 73.232  |
| 80 (onnx fp16)                | 66.862  |
| 20 (trt fp16+layernorm plugin)| 90.582  |
| 40 (trt fp16+layernorm plugin)| 75.411  |
| 60 (trt fp16+layernorm plugin)| 69.602  |
| 80 (trt fp16+layernorm plugin)| 65.603  |