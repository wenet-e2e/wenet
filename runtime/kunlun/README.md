# 在昆仑芯片上运行Wenet
## 介绍
下面的示例展示了如何在XPU上部署WeNet离线或在线的ASR模型。XPU是一种由昆仑芯100%自主研发的通用人工智能计算核心架构。

## 准备XPU运行环境

在开始之前，请确认您获得以下必须的环境。

    XRE(XPU Runtime Environment):昆仑芯片的基础运行环境，包括芯片驱动程序、runtime api库、固件FW工具等功能模块。
    XDNN(XPU Deep Neural Network Library):加速深度神经网络的昆仑芯片库，提供应用程序中使用的高性能DNN功能库。

如果您需要任何帮助，或是想要进一步了解昆仑芯片，请通过官方网址联系我们：
https://www.kunlunxin.com.cn/

## 操作步骤
- 第一步：构建，需要cmake 3.14及以上版本

``` sh
export CXX=${your_g++_path}
export CC=${your_gcc_path}
export XPU_API_PATH=${your_api_path}

# -r : release version; -d : debug version
bash ./compile.sh -r
```

- 第二步：测试，测试结果将在控制台输出

``` sh
## set KUNLUN XPU visible device
export XPU_VISIBLE_DEVICES=0
export XPUSIM_DEVICE_MODEL=KUNLUN2
## set logging level
export GLOG_logtostderr=1
export GLOG_v=3
## set speech wav and model/weight path
wav_path=${your_test_wav_path}
xpu_model_dir=${your_xpu_weight_dir}
units=${your_units.txt}
## executive command
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path ${wav_path} \
    --xpu_model_dir ${xpu_model_di} \
    --unit_path ${units}   \
    --device_id 0           \
    --nbest  3  2>&1 | tee log.txt
```

单条语音执行结果如下所示:

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
