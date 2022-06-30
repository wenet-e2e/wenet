# Runtime on WeNet

This is the runtime of WeNet.

We are going to support the following platforms:

1. Various deep learning inference engines, such as LibTorch, ONNX, OpenVINO, TVM and so on.
2. Various OS, such as android, iOS, Harmony and so on.
3. Various AI chips, such as GPU, Horzion BPU and so on.
4. Various hardware platforms, such as Raspberry Pi.
5. Various language bingding, such as python and go.

Feel free to volunteer yourself if you are interested in trying out some items(they do not have to be on the list).

## Progress

For each platform, we will create a subdirectory on in runtime. Currently, we have:

- [x] LibTorch: in c++, the default engine of WeNet.
- [x] OnnxRuntime: in c++, the official runtime for onnx model.
- [x] GPU: in python, powered by triton.
- [x] android: in java, it shows an APP demo.
- [ ] Language binding
  - [x] bingding/python: python is the first class for binding.
  - [ ] bingding/go: ongoing.


Please note `server` and `device` is depreated, it will be removed in the future.
