# x86 平台上使用 WeNet 进行语音识别

Wenet 基于 pytorch 框架进行语音识别模型训练，而在使用训练好的 Wenet 模型进行真实场景的语音识别任务时，需要更高效的执行效率和一些外围组件。因此我们提供了一套基于 C++ 实现的 Wenet 的语音识别工具和在线服务。


## 使用docker启动语音识别服务

最简单的使用 Wenet 的方式是通过官方提供的 docker 镜像 `wenetorg/wenet:mini` 来启动服务。

下面的命令先下载官方提供的预训练模型，并启动 docker 服务，加载模型，提供 websocket 协议的语音识别服务。
``` sh
cd wenet/runtime/server/x86
wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_libtorch.tar.gz
tar -xf 20210602_u2++_conformer_libtorch.tar.gz
model_dir=$PWD/20210602_u2++_conformer_libtorch
docker run --rm -it -p 10086:10086 -v $model_dir:/home/wenet/model wenetorg/wenet-mini:latest bash /home/run.sh
```

`$model_dir` 是模型在本机的目录，将被映射到容器的 `/home/wenet/model` 目录，然后启动 web 服务。

**实时识别**

使用浏览器打开文件`web/templates/index.html`，在 `WebSocket URL：`填入 `ws://127.0.0.1:10086`, 允许浏览器弹出的请求使用麦克风，即可通过麦克风进行实时语音识别。

![Runtime web](../../../docs/images/runtime_web.png)

## 自行编译运行时程序

如果想使用非 docker 方式，需要自行编译。Wenet 支持 linux/macos/windows 三种平台上的编译。需要安装 cmake 3.14 或者更高版本。

运行如下命令，完成编译。

``` sh
# 当前目录为 wenet/runtime/server/x86
mkdir build && cd build && cmake .. && cmake --build .
```
或者使用命令编译以支持 gRPC。

``` sh
mkdir build && cd build && cmake -DGRPC=ON .. && cmake --build .
```

编译好的可执行程序在 `wenet/runtime/server/x86/build/` 下：

* decoder_main 本地文件识别工具
* websocket_server_main 基于websocket协议的识别服务端
* websocket_client_main 基于websocket协议的识别客户端


下载预训练模型

``` sh
# 当前目录为 wenet/runtime/server/x86
wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_libtorch.tar.gz
tar -xf 20210602_u2++_conformer_libtorch.tar.gz
```

## 本地wav文件识别

本地文件识别，即程序每次运行时，给定一个语音文件或者一组语音文件列表，输出识别结果，然后结束程序。

下载好模型后，执行如下的命令进行本地wav文件识别，将 `wav_path` 设为你想测试的 wav 文件地址，将 `model_dir` 设为你的模型目录地址。

``` sh
# 当前目录为 wenet/runtime/server/x86
# 已经下载并解压20210602_unified_transformer_server.tar.gz到当前目录
# 准备好一个16k采样率，单通道，16bits的音频文件test.wav

export GLOG_logtostderr=1
export GLOG_v=2
wav_path=test.wav
model_dir=./20210602_unified_transformer_server
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```

`decoder_main`工具支持两种wav文件模式：
 * 使用`--wav_path`指定单个文件，一次识别单个wav文件。
 * 使用`--wav_scp`指定一个.scp格式的wav列表，一次识别多个wav文件。

执行 `./build/bin/decoder_main --help`  可以了解更多的参数意义。

## 基于websocket的在线识别服务

在线识别服务，即程序运行后会常驻在内存中，等待客户端的请求，对于客户端发来的语音数据进行识别，将识别文本返回给客户端。

在这个示例中，需要先启动服务端程序，然后再启动客户端发送请求。

### 启动websocket识别服务端

执行如下指令，将 `model_dir` 设置为你的模型目录地址。

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=./20210602_unified_transformer_server
./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```

上述服务启动后，会监听 10086 端口。若想使用其他端口，请修改 `--port` 对应的参数.

### websocket 识别客户端

客户端按 websocket 协议去请求服务，可以用不同语言来实现客户端。我们提供了两种客户端，一种是基于 C++ 的命令行工具。一种是基于网页形式的可视化客户端。

**命令行 websocket 客户端**

打开一个新的命令行窗口，运行如下指令，启动客户端。可将 `wav_path` 设为你想测试的 wav 文件地址。

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=test.wav
./build/bin/websocket_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

该程序会模拟语音数据的真实时间进行流式请求，即 10 秒的语音会按 10 秒时间发送完。可以在客户端和服务器端看到流式识别过程输出的信息。

![Runtime server demo](../../../docs/images/runtime_server.gif)

注意 `--port` 需要设置为服务端使用的端口号。

如果有两台机器，也可以在一台机器上运行服务端，在另一台机器运行客户端，此时 `--hostname` 要指定为服务端所在机器的可访问 ip。

**网页版 websocket 客户端**

网页版客户端支持麦克风的语音输入。

使用浏览器打开文件 `web/templates/index.html`, 在 `Websoket URL` 里设置 websoket 识别服务的地址，比如 `ws://localhost:10086`, 点击开始识别。

**时延信息计算**

`server.log` 文件中记录了每次请求的时延，可以通过如下命令得到所有请求的平均时延。

``` sh
grep "Rescoring cost latency" server.log | awk '{sum += $NF}; END {print sum/NR}'
```

## 在 Docker 环境中使用

如果遇到问题比如无法编译，我们提供了 docker 镜像用于直接执行示例。需要先安装好 docker，运行如下命令，进入 docker 容器环境。

``` sh
docker run --rm -it mobvoiwenet/wenet:latest bash
```

该镜像包含了编译过程中所依赖的所有第三方库、编译好的文件和预训练模型。

预训练模型在 `/home/model` 目录, 可执行程序在 `/home/wenet/runtime/server/x86/build` 目录。

### 构建 Docker 镜像

我们也提供了 Dockerfile，可以自己构建 docker 镜像，参考 `docker/Dockerfile` 文件。

``` sh
cd docker
docker build --no-cache -t wenet:latest .
docker run --rm -it wenet bash
```
