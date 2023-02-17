# OpenVINO™ backend on WeNet

* Step 1. Get ONNX model.

Use `wenet.bin.export_onnx_cpu` to export ONNX model. Follow this [guide](https://github.com/wenet-e2e/wenet/blob/main/runtime/onnxruntime/README.md)

Please note, 3 ONNX models(`encoder.onnx`, `ctc.onnx`, and `decoder.onnx`) will be generated.

* Step 2. Convert ONNX model to OpenVINO™ IR(Intermediate Representation).

``` sh
mo --input_model onnx/encoder.onnx --input chunk,att_cache,cnn_cache --input_shape [1,-1,80],[12,4,-1,128],[12,1,256,7] --output_dir openvino 
mo --input_model onnx/ctc.onnx --input_shape [1,-1,256] --output_dir openvino 
mo --input_model onnx/decoder.onnx --input hyps,hyps_lens,encoder_out --input_shape [-1,-1],[-1],[1,-1,256] --output_dir openvino

# When it finishes, you can find IR files(.xml and .bin) for encoder, ctc and decoder.
```

* Step 3. Build WeNet with OpenVINO™. 

Please refer [system requirement](https://github.com/openvinotoolkit/openvino#system-requirements) to check if the hardware platform available by OpenVINO™. 

``` sh
mkdir build && cd build
cmake -DOPENVINO=ON -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF ..
cmake --build .
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
