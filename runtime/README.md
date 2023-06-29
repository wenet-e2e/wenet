# Runtime on WeNet

This is the runtime of WeNet.

We are going to support the following platforms:

1. Various deep learning inference engines, such as LibTorch, ONNX, OpenVINO, TVM, and so on.
2. Various OS, such as android, iOS, Harmony, and so on.
3. Various AI chips, such as GPU, Horzion BPU, and so on.
4. Various hardware platforms, such as Raspberry Pi.
5. Various language binding, such as python and go.

Feel free to volunteer yourself if you are interested in trying out some items(they do not have to be on the list).

## Introduction

Here is a brief summary of all platforms and OSs. please note the corresponding working `OS` and `inference engine`.

| runtime         | OS                  | inference engine     | Description                                                                                      |
|-----------------|---------------------|----------------------|--------------------------------------------------------------------------------------------------|
| core            | /                   | /                    | common core code of all runtime                                                                  |
| android         | android             | libtorch             | android demo, [English demo](https://www.youtube.com/shorts/viEnvmZf03s ), [Chinese demo](TODO)  |
| bingding/python | linux, windows, mac | libtorch             | python binding of wenet, mac M1/M2 are is not supported now.                                     |
| gpu             | linux               | onnxruntime/tensorrt | GPU inference with NV's Triton and TensorRT                                                      |
| horizonbpu      | linux               | bpu runtime          | Horizon BPU runtime                                                                              |
| ios             | ios                 | libtorch             | ios demo, [link](TODO)                                                                           |
| kunlun          | linux               | xpu runtime          | Kunlun XPU runtime                                                                               |
| libtorch        | linux, windows, mac | libtorch             | c++ build with libtorch                                                                          |
| ipex            | linux               | libtorch + ipex      | c++ build with libtorch and ipex optimization                                                                         |
| onnxrutnime     | linux, windows, mac | onnxruntime          | c++ build with onnxruntime                                                                       |
| openvino        | linux, windows, mac | openvino             | c++ build with openvino                                                                          |
| raspberrypi     | linux               | onnxruntime          | c++ build on raspberrypi with onnxruntime                                                        |
| web             | linux, windows, mac | libtorch             | web demo with gradio and python binding, [link]()                                                |

