# WeNet Server (x86) ASR Demo With Intel® Extension for PyTorch\* Optimization

[Intel® Extension for PyTorch\*](https://github.com/intel/intel-extension-for-pytorch) (IPEX) extends [PyTorch\*](https://pytorch.org/) with up-to-date  optimization features for extra performance boost on Intel hardware. The optimizations take advantage of AVX-512, Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs.

In the following we are introducing how to accelerate WeNet model inference performance on Intel® CPU machines with the adoption of Intel® Extension for PyTorch\*. The adoption mainly includes the export of pretrained models with IPEX optimization, as well as the buildup of WeNet runtime executables with IPEX C++ SDK.

## Run with Prebuilt Docker (TBD, current is stock libtorch)

* Step 1. Download pretrained model(see the following link) or prepare your trained model.

[中文(WenetSpeech)](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz)
| [English(GigaSpeech)](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/gigaspeech_u2pp_conformer_libtorch.tar.gz)


* Step 2. Start docker websocket server. Here is a demo.

``` sh
model_dir=$PWD/20210602_u2++_conformer_libtorch  # absolute path
docker run --rm -it -p 10086:10086 -v $model_dir:/home/wenet/model wenetorg/wenet-mini:latest bash /home/run.sh
```

* Step 3. Test with web browser. Open runtime/libtorch/web/templates/index.html in the browser directly, input your `WebSocket URL`, it will request some permissions, and start to record to test, as the following graph shows.

![Runtime web](../../docs/images/runtime_web.png)

## Run in Docker Build (TBD, current is stock libtorch)

We recommend using the docker environment to build the c++ binary to avoid
system and environment problems.

* Step 1. Build your docker image.

``` sh
cd docker
docker build --no-cache -t wenet:latest .
```

* Step 2. Put all the resources, like model, test wavs into a docker resource dir.

``` sh
mkdir -p docker_resource
cp -r <your_model_dir> docker_resource/model
cp <your_test_wav> docker_resource/test.wav
```

* Step 3. Start docker container.
``` sh
docker run --rm -v $PWD/docker_resource:/home/wenet/runtime/libtorch/docker_resource -it wenet bash
```

* Step 4. Testing in docker container
```
cd /home/wenet/runtime/libtorch
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=docker_resource/test.wav
model_dir=docker_resource/model
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```

Or you can do the WebSocket server/client testing as described in the `WebSocket` section.

## Run with Local Build

* Step 1. Environment Setup.

WeNet code cloning and default dependencies installation
``` sh
git clone https://github.com/wenet-e2e/wenet
cd wenet
pip install -r requirements.txt
```
Installation of IPEX
``` sh
pip install intel_extension_for_pytorch==1.13.100 -f https://developer.intel.com/ipex-whl-stable-cpu
```

Installation of related tools: Intel® OpenMP and TCMalloc

`pip install intel-openmp`

and

`yum -y install gperftools` or `apt-get install -y google-perftools`

based on the package manager of your system.

* Step 2. Download or prepare your pretrained model.

* Step 3. Export the pretrained model with IPEX optimization.

For exporting FP32 runtime model
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_file <runtime_model_output_filename>
```
If you have an Intel® 4th Generation Xeon (Sapphire Rapids) server, you can export a BF16 runtime model and get better performance by virtue of AMX instructions
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_file <runtime_model_output_filename> \
    --dtype bf16
```
And for exporting int8 quantized runtime model
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_quant_file <runtime_model_output_filename>
```

* Step 4. Build runtime executable binaries. The build requires cmake 3.14 or above. For building, please first change to `wenet/runtime/ipex` as your build directory, then type:

``` sh
mkdir build && cd build && cmake .. && cmake --build .
```

* Step 5. Testing, the RTF(real time factor) would be shown in the console.

``` sh
cd ..
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
model_dir=your_model_dir
ipexrun --no_python \
    ./build/bin/decoder_main \
        --chunk_size -1 \
        --wav_path $wav_path \
        --model_path $model_dir/final.zip \
        --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```
NOTE: Please refer [IPEX Launch Script Usage Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/1.13.100+cpu/tutorials/performance_tuning/launch_script.html) for usage of advanced features.

## Advanced Usage (TBD, change to ipexrun if not to be deleted)

### WebSocket

* Step 1. Download or prepare your pretrained model.
* Step 2. Build as in `Run with Local Build`
* Step 3. Start WebSocket server.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```
* Step 4. Start WebSocket client.

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/bin/websocket_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

You can also start WebSocket client by web browser as described before.

Here is a demo for command line based websocket server/client interaction.

![Runtime server demo](../../../docs/images/runtime_server.gif)

### gRPC

Why grpc? You may find your answer in https://grpc.io/.
Please follow the following steps to try gRPC.

* Step 1. Download or prepare your pretrained model.
* Step 2. Build
``` sh
mkdir build && cd build && cmake -DGRPC=ON .. && cmake --build .
```
* Step 3. Start gRPC server

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/bin/grpc_server_main \
    --port 10086 \
    --workers 4 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```

* Step 4. Start gRPC client.

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/bin/grpc_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

### http

* Step 1. Download or prepare your pretrained model.
* Step 2. Build
``` sh
mkdir build && cd build && cmake -DHTTP=ON .. && cmake --build .
```
* Step 3. Start http server

simply replace grpc_server_main with http_server_main of Step 3 in gRPC

* Step 4. Start http client.

simply replace grpc_client_main with http_client_main of Step 4 in gRPC
