# ONNX backend on WeNet

* Step 1. Export your experiment model to ONNX by https://github.com/wenet-e2e/wenet/blob/main/wenet/bin/export_onnx_cpu.py

``` sh
exp=exp  # Change it to your experiment dir
onnx_dir=onnx
python -m wenet.bin.export_onnx_cpu \
  --config $exp/train.yaml \
  --checkpoint $exp/final.pt \
  --chunk_size 16 \
  --output_dir $onnx_dir \
  --num_decoding_left_chunks -1

# When it finishes, you can find `encoder.onnx`, `ctc.onnx`, and `decoder.onnx` in the $onnx_dir respectively.
```

* Step 2. Build. The build requires cmake 3.14 or above.

``` sh
mkdir build && cd build
cmake -DONNX=ON -DTORCH=OFF -DWEBSOCET=OFF -DGRPC=OFF ..
cmake --build .
```

* Step 3. Testing, the RTF(real time factor) is shown in the console.

``` sh
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
