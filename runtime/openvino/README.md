# OpenVINO™ backend on WeNet

* Step 1. Get ONNX model.

Use `wenet.bin.export_onnx_cpu` to export ONNX model. Follow this [guide](https://github.com/wenet-e2e/wenet/blob/main/runtime/onnxruntime/README.md)

Please note, 3 ONNX models(`encoder.onnx`, `ctc.onnx`, and `decoder.onnx`) will be generated.

* Step 2. Convert ONNX model to OpenVINO™ IR(Intermediate Representation).

Install OpenVINO™ runtime with Model Optimizer (MO) command tool for model conversion:

``` sh
mo --input_model onnx/encoder.onnx --input chunk[1,-1,80],att_cache[12,4,-1,128],cnn_cache[12,1,256,7] --output_dir openvino 
mo --input_model onnx/ctc.onnx --input_shape [1,-1,256] --output_dir openvino 
mo --input_model onnx/decoder.onnx --input hyps[-1,-1],hyps_lens[-1],encoder_out[1,-1,256]  --output_dir openvino

By using MO, user can specify shape of multiple inputs by `--input` option, or just directly provide `--input_shape` for single input model. To enable the dynamic shape support for inference, user can use `-1` or provide a range of input shape value, like `1..80`.Please refer the [usage guide of MO](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).


# When it finishes, you can find IR files(.xml and .bin) for encoder, ctc and decoder.
```

* Step 3. Build WeNet with OpenVINO™.

Please refer [system requirement](https://github.com/openvinotoolkit/openvino#system-requirements) to check if the hardware platform available by OpenVINO™.
The OpenVINO Linux installation package will not provide Intel TBB runtime library and pugixml library, please check if your system already installed:
``` sh
# For Ubuntu and Debian:
sudo apt install libtbb-dev libpugixml-dev
# For CentOS:
sudo yum install tbb-devel pugixml-devel
```
To build WeNet with OpenVINO:
``` sh
mkdir build && cd build
cmake -DOPENVINO=ON -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF ..
make --jobs=$(nproc --all)
```

(Optional) Some users may cannot easily download OpenVINO™ binary package from server due to firewall or proxy issue. If you failed to download by CMake script, you can download OpenVINO™ package by your selves and put the package to below path:

``` sh
${wenet_path}/runtime/openvino/fc_base/openvino-subbuild/openvino-populate-prefix/src/l_openvino_toolkit_ubuntu20_2022.3.0.9052.9752fafe8eb_x86_64.tgz
```

* Step 4. Testing.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
openvino_dir=your_model_dir
units=units.txt  # Change it to your model units path
./build/bin/decoder_main \
    --chunk_size 16 \
    --wav_path $wav_path \
    --openvino_dir $openvino_dir \
    --unit_path $units 2>&1 | tee log.txt
```
