# WeNet On-device ASR Android Demo

This Android demo shows we can run on-device streaming ASR with WeNet. You can download our prebuilt APK or build your APK from source code.

## Prebuilt APK

* [Chinese ASR Demo APK, with model trained on AIShell data](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210202_app.apk)
* [English ASR Demo APK, with model trained on LibriSpeech data (To be added)]()

## Build your APK from source code

### 1) Build model

You can use our pretrained model (click the following link to download):

* [Chinese model trained on AIshell](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210121_unified_transformer_device.tar.gz)
* [English model trained on LibriSpeech(TODO)]()

Or you can train your own model using WeNet training pipeline on your data.

### 2) Build APK

When your model is ready, put `final.zip` and `words.txt` into Android assets (`app/src/main/assets`) folder,
then just build and run the APK. Here is a gif demo, which shows how our on-device streaming e2e ASR runs with low latency.
Please note the wifi and data has been disabled in the demo so there is no network connection ^\_^.

![Runtime android demo](../../../../docs/images/runtime_android.gif)
