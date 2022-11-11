# WeNet & Horizon BPU (Cross Compile)

* Step 1. Install cross compile tools in the PC.

``` sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Or download, and install the binaries from: https://releases.linaro.org/components/toolchain/binaries/latest-7


* Step 2. Export your experiment model to ONNX by https://github.com/wenet-e2e/wenet/blob/main/wenet/bin/export_onnx_bpu.py

``` sh
exp=exp  # Change it to your experiment dir
onnx_dir=onnx
python -m wenet.bin.export_onnx_bpu \
  --config $exp/train.yaml \
  --checkpoint $exp/final.pt \
  --chunk_size 8 \
  --output_dir $onnx_dir \
  --num_decoding_left_chunks 4

# When it finishes, you can find `encoder.onnx`, and `ctc.onnx` in the $onnx_dir respectively.
```

* Step 3. Build. The build requires cmake 3.14 or above. and Send the binary and libraries to Horizon X3PI.

``` sh
cmake -B build -DHORIZONBPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build
scp build/bin/decoder_main sunrise@xxx.xxx.xxx:/path/to/wenet
scp fc_base/easydnn-src/lib/libeasydnn.so* sunrise@xxx.xxx.xxx:/path/to/wenet
```

* Step 4. Testing, the RTF(real time factor) is shown in Horizon X?PI's console.

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
