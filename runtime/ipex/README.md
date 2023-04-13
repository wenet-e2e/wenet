## WeNet Server (x86) ASR Demo With Intel® Extension for PyTorch\* Optimization

[Intel® Extension for PyTorch\*](https://github.com/intel/intel-extension-for-pytorch) (IPEX) extends [PyTorch\*](https://pytorch.org/) with up-to-date  optimization features for extra performance boost on Intel hardware. The optimizations take advantage of AVX-512, Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs.

In the following we are introducing how to accelerate WeNet model inference performance on Intel® CPU machines with the adoption of Intel® Extension for PyTorch\*. The adoption mainly includes the export of pretrained models with IPEX optimization, as well as the buildup of WeNet runtime executables with IPEX C++ SDK.



* Step 1. Environment Setup.

WeNet code cloning and default dependencies installation
``` sh
git clone https://github.com/wenet-e2e/wenet
cd wenet
pip install -r requirements.txt
```
Installation of IPEX
``` sh
pip install intel_extension_for_pytorch==1.13.100 -f https://developer.intel.com/ipex-whl-stable-cpu
```

Installation of related tools: Intel® OpenMP and TCMalloc

`pip install intel-openmp`

and

`yum -y install gperftools` or `apt-get install -y google-perftools`

based on the package manager of your system.

* Step 2. Download or prepare your pretrained model.

* Step 3. Export the pretrained model with IPEX optimization.

For exporting FP32 runtime model
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_file <runtime_model_output_filename>
```
If you have an Intel® 4th Generation Xeon (Sapphire Rapids) server, you can export a BF16 runtime model and get better performance by virtue of AMX instructions
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_file <runtime_model_output_filename> \
    --dtype bf16
```
And for exporting int8 quantized runtime model
``` sh
source examples/aishell/s0/path.sh
python wenet/bin/export_ipex.py \
    --config <model_config_yaml_filename> \
    --checkpoint <model_ckpt_filename> \
    --output_quant_file <runtime_model_output_filename>
```

* Step 4. Build runtime executable binaries. The build requires cmake 3.14 or above. For building, please first change to `wenet/runtime/ipex` as your build directory, then type:

``` sh
mkdir build && cd build && cmake .. && cmake --build .
```

* Step 5. Testing, the RTF(real time factor) would be shown in the console.

``` sh
cd ..
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
model_dir=your_model_dir
ipexrun --no_python \
    ./build/bin/decoder_main \
        --chunk_size -1 \
        --wav_path $wav_path \
        --model_path $model_dir/final.zip \
        --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```
NOTE: Please refer [IPEX Launch Script Usage Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/1.13.100+cpu/tutorials/performance_tuning/launch_script.html) for usage of advanced features.

For advanced usage of WeNet, such as building Web/RPC/HTTP services, please refer [LibTorch Tutorial](../libtorch#advanced-usage). The difference is that the executables should be invoked via IPEX launch script `ipexrun`.