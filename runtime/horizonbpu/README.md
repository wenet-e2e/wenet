# WeNet & Horizon BPU (Cross Compile)

* Step 1. Install horizon packages and cross compile tools in the PC.

NOTE: Make sure you have installed WeNet conda environment, see https://github.com/wenet-e2e/wenet#installationtraining-and-developing

```sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
wget https://github.com/xingchensong/toolchain_pkg/releases/download/ai_toolchain/wheels.tar.gz
tar -xzf wheels.tar.gz
conda activate wenet
pip install wheels/* -i https://mirrors.aliyun.com/pypi/simple
pip install onnx onnxruntime -i https://mirrors.aliyun.com/pypi/simple
```


* Step 2. Export model to ONNX and convert ONNX to Horizon .bin (~40min)


``` sh
export WENET_DIR=$PWD/../../
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
wget model_subsample8_parameter110M.tar.gz
tar -xzf model_subsample8_parameter110M.tar.gz

python3 $WENET_DIR/tools/onnx2horizonbin.py \
  --config ./model_subsample8_parameter110M/train.yaml \
  --checkpoint ./model_subsample8_parameter110M/final.pt \
  --output_dir ./model_subsample8_parameter110M/sample50_chunk4_leftchunk32 \
  --chunk_size 4 \
  --num_decoding_left_chunks 32 \
  --max_samples 50 \
  --dict ./model_subsample8_parameter110M/units.txt \
  --cali_datalist ./model_subsample8_parameter110M/calibration_data/data.list
```

* Step 3. Build. The build requires cmake 3.14 or above. and Send the binary and libraries to Horizon X3PI.

``` sh
cmake -B build -DBPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build
export BPUIP=xxx.xxx.xxx
export DEMO_PATH_ON_BOARD=/path/to/demo
scp build/bin/decoder_main sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/dnn/*j3*/*/*/lib/libdnn.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/easy_dnn/*j3*/*/*/lib/libeasy_dnn.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/hlog/*j3*/*/*/lib/libhlog.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
```

* Step 4. Testing, the RTF(real time factor) is shown in Horizon X3PI's console.

``` sh
cd /path/to/demo
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
bpu_model_dir=your_model_dir
units=your_dict_path
./decoder_main \
    --chunk_size 4 \
    --num_left_chunks 32 \
    --rescoring_weight 0.0 \
    --wav_path $wav_path \
    --bpu_model_dir $bpu_model_dir \
    --unit_path $units 2>&1 | tee log.txt
```
