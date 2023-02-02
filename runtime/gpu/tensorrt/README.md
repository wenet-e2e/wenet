### Using Tensorrt for Triton ASR Server

```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
docker pull soar97/triton-wenet:22.12
docker run -it --rm --name "wenet_trt_test" --gpus all --shm-size 1g --net host soar97/triton-wenet:22.12
# inside the docker container
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/runtime/gpu/tensorrt
pip3 install nvidia-pyindex
pip3 install -r requirements.txt

bash run.sh
```
