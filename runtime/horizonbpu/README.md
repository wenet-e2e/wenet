# WeNet & Horizon BPU (Cross Compile)

* Step 1. Setup environment (install horizon packages and cross compile tools) in the PC. (~10min)

```sh
# Conda env (This conda env is only used for converting bpu models, not for training torch models,
#   It's OK to install cpu-version pytorch)
conda create -n horizonbpu python=3.8
conda activate horizonbpu
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/runtime/horizonbpu
pip install -r ../../requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install torch==1.13.0 torchaudio==0.13.0 torchvision==0.14.0 onnx onnxruntime -i https://mirrors.aliyun.com/pypi/simple

# Horizon packages
wget https://gitee.com/xcsong-thu/toolchain_pkg/releases/download/resource/wheels.tar.gz
tar -xzf wheels.tar.gz
pip install wheels/* -i https://mirrors.aliyun.com/pypi/simple

# Cross compile tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```


* Step 2. Build decoder_main. It requires cmake 3.14 or above. and Send the binary/libraries to Horizon X3PI. (~20min)

``` sh
# Assume current dir is `wenet/runtime/horizonbpu`
cmake -B build -DBPU=ON -DONNX=OFF -DTORCH=OFF -DWEBSOCKET=OFF -DGRPC=OFF -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build

# Send binary and libraries
export BPUIP=xxx.xxx.xxx.xxx
export DEMO_PATH_ON_BOARD=/path/to/demo
scp build/bin/decoder_main sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/dnn/*j3*/*/*/lib/libdnn.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/easy_dnn/*j3*/*/*/lib/libeasy_dnn.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp fc_base/easy_dnn-src/hlog/*j3*/*/*/lib/libhlog.so sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
```

* Step 3. Export model to ONNX and convert ONNX to Horizon .bin and Send the model/dict/test_wav to Horizon X3PI. (~40min)

``` sh
# Assume current dir is `wenet/runtime/horizonbpu`
conda activate horizonbpu
export WENET_DIR=$PWD/../../
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# Download torch model
wget https://ghproxy.com/https://github.com/xingchensong/toolchain_pkg/releases/download/conformer_subsample8_110M/model_subsample8_parameter110M.tar.gz
tar -xzf model_subsample8_parameter110M.tar.gz

# Convert torch model to bpu model (*.pt -> *.onnx -> *.bin)
# NOTE(xcsong): Convert model with 110M parameters requires CPU MEM >= 16G,
#   if your CPU does not meet the requirement, you can download pre-converted encoder.bin/ctc.bin
#   via this link: https://github.com/xingchensong/toolchain_pkg/releases
python3 $WENET_DIR/tools/onnx2horizonbin.py \
  --config ./model_subsample8_parameter110M/train.yaml \
  --checkpoint ./model_subsample8_parameter110M/final.pt \
  --output_dir ./model_subsample8_parameter110M/sample50_chunk8_leftchunk16 \
  --chunk_size 8 \
  --num_decoding_left_chunks 16 \
  --max_samples 50 \
  --dict ./model_subsample8_parameter110M/units.txt \
  --cali_datalist ./model_subsample8_parameter110M/calibration_data/data.list

# scp test wav file and dictionary
scp ./model_subsample8_parameter110M/test_wav.wav sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp ./model_subsample8_parameter110M/units.txt sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
# scp bpu models
scp ./model_subsample8_parameter110M/sample50_chunk8_leftchunk16/hb_makertbin_output_encoder/encoder.bin sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
scp ./model_subsample8_parameter110M/sample50_chunk8_leftchunk16/hb_makertbin_output_ctc/ctc.bin sunrise@$BPUIP:$DEMO_PATH_ON_BOARD
```

* Step 4. Testing on X3PI, the RTF(real time factor) is shown in Horizon X3PI's console. (~1min)

``` sh
cd /path/to/demo
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH \
  GLOG_logtostderr=1 GLOG_v=2 \
  ./decoder_main \
      --chunk_size 8 \
      --num_left_chunks 16 \
      --rescoring_weight 0.0 \
      --wav_path ./test_wav.wav \
      --bpu_model_dir ./ \
      --unit_path ./units.txt 2>&1 | tee log.txt
```
