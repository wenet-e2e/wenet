# WeNet & Raspberry PI (Cross Compile)

* Step 1. Install cross compile tools in the PC.

``` sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Or download, and install the binaries from: https://releases.linaro.org/components/toolchain/binaries/latest-7


* Step 2. Export your experiment model to ONNX by https://github.com/wenet-e2e/wenet/blob/main/wenet/bin/export_onnx_cpu.py

``` sh
exp=exp  # Change it to your experiment dir
onnx_dir=onnx
python -m wenet.bin.export_onnx_cpu \
  --config $exp/train.yaml \
  --checkpoint $exp/final.pt \
  --chunk_size 16 \
  --output_dir $onnx_dir \
  --num_decoding_left_chunks -1

# When it finishes, you can find `encoder.onnx(.quant)`, `ctc.onnx(.quant)`, and `decoder.onnx(.quant)` in the $onnx_dir respectively.
# We use the quantified to speed up the inference, so rename it without the suffix `.quant`
```

* Step 3. Build. The build requires cmake 3.14 or above. and Send the binary and libraries to Raspberry PI.

``` sh
cmake -B build -DONNX=ON -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build
scp build/bin/decoder_main pi@xxx.xxx.xxx:/path/to/wenet
scp fc_base/onnxruntime-src/lib/libonnxruntime.so* pi@xxx.xxx.xxx:/path/to/wenet
```

* Step 4. Testing, the RTF(real time factor) is shown in Raspberry PI's console.

``` sh
cd /path/to/wenet
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
onnx_dir=your_model_dir
units=units.txt  # Change it to your model units path
./build/bin/decoder_main \
    --chunk_size 16 \
    --wav_path $wav_path \
    --onnx_dir $onnx_dir \
    --unit_path $units 2>&1 | tee log.txt
```
