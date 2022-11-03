# WeNet running on KUNLUNXIN XPU device
## Introduction
The below example shows how to deploy WeNet offline and online ASR models on XPUs.
XPU is a core architecture 100% independently developed by KUNLUNXIN for general artificial intelligence computing.

## Setup environment for XPU device

Before the start, makesure you have these necessary environment

    XRE(XPU Runtime Environment):The basic operating environment of the XPUs
    includes functional modules such as chip drivers, runtime api library, and firmware tools.

    XDNN(XPU Deep Neural Network Library): XPU library for accelerating deep neural networks, providing high-performance DNN function library used in applications.

If you would like to know more about XPUs or need any help, please contact us through the official website:

https://www.kunlunxin.com.cn/

## Instruction
- Step 1. Build, the build requires cmake 3.14 or above.

``` sh
export CXX=${your_g++_path}
export CC=${your_gcc_path}
export XPU_API_PATH=${your_api_path}

# -r : release version; -d : debug version
bash ./compile.sh -r
```

- Step 2. Testing, the result is shown in the console.

``` sh
## set KUNLUN XPU visible device
export XPU_VISIBLE_DEVICES=0
export XPUSIM_DEVICE_MODEL=KUNLUN2
## set logging level
export GLOG_logtostderr=1
export GLOG_v=3
## set speech wav and model/weight/units path
wav_path=${your_test_wav_path}
xpu_model_dir=${your_xpu_weight_dir}
units=${your_units.txt}
## executive command
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --xpu_model_dir $xpu_model_dir \
    --unit_path $units   \
    --device_id 0           \
    --nbest  3  2>&1 | tee log.txt
```

A typical output result is as following:

``` sh
XPURT /docker_workspace/icode-api/baidu/xpu/api/../runtime/output/so/libxpurt.so loaded
I1027 06:06:21.933722 111767 params.h:152] Reading XPU WeNet model weight from /docker_workspace/icode-api/baidu/xpu/api/example/wenet-conformer/all_data/
I1027 06:06:21.934103 111767 xpu_asr_model.cc:46] XPU weight_dir is: /docker_workspace/icode-api/baidu/xpu/api/example/wenet-conformer/all_data//model_weights/
I1027 06:06:23.832731 111767 xpu_asr_model.cc:65] ======= XPU Kunlun Model Info: =======
I1027 06:06:23.832749 111767 xpu_asr_model.cc:66]       subsampling_rate 4
I1027 06:06:23.832777 111767 xpu_asr_model.cc:67]       right_context 6
I1027 06:06:23.832789 111767 xpu_asr_model.cc:68]       sos 5538
I1027 06:06:23.832795 111767 xpu_asr_model.cc:69]       eos 5538
I1027 06:06:23.832799 111767 xpu_asr_model.cc:70]       is bidirectional decoder 1
I1027 06:06:23.832804 111767 params.h:165] Reading unit table /docker_workspace/icode-api/baidu/xpu/api/example/wenet-conformer/all_data/dict
I1027 06:06:23.843475 111776 decoder_main.cc:54] num frames 418
I1027 06:06:23.843521 111776 asr_decoder.cc:104] Required 2147483647 get 418
I1027 06:06:23.843528 111776 xpu_asr_model.cc:116] Now Use XPU:0!
I1027 06:06:23.843616 111776 xpu_asr_model.cc:173]       max_seqlen is 418
I1027 06:06:23.843619 111776 xpu_asr_model.cc:174]       q_seqlen   is 103
I1027 06:06:23.843623 111776 xpu_asr_model.cc:175]       att_dim    is 512
I1027 06:06:23.843626 111776 xpu_asr_model.cc:176]       ctc_dim    is 5538
I1027 06:06:23.852284 111776 asr_decoder.cc:113] forward takes 7 ms, search takes 1 ms
I1027 06:06:23.852383 111776 asr_decoder.cc:194] Partial CTC result 甚至出现交易几乎停滞的情况
I1027 06:06:23.852530 111776 asr_decoder.cc:194] Partial CTC result 甚至出现交易几乎停滞的情况
I1027 06:06:23.852537 111776 xpu_asr_model.cc:248]       num_hyps  is 3
I1027 06:06:23.852541 111776 xpu_asr_model.cc:249]       beam_size is 3
I1027 06:06:23.852545 111776 xpu_asr_model.cc:250]       new_bs    is 3
I1027 06:06:23.852545 111776 xpu_asr_model.cc:251]       max_hyps_len is 14
I1027 06:06:23.853902 111776 asr_decoder.cc:84] Rescoring cost latency: 1ms.
I1027 06:06:23.853911 111776 decoder_main.cc:72] Partial result: 甚至出现交易几乎停滞的情况
I1027 06:06:23.853914 111776 decoder_main.cc:104] test Final result: 甚至出现交易几乎停滞的情况
I1027 06:06:23.853924 111776 decoder_main.cc:105] Decoded 4203ms audio taken 10ms.
test 甚至出现交易几乎停滞的情况
I1027 06:06:23.853984 111767 decoder_main.cc:180] Total: decoded 4203ms audio taken 10ms.
```
