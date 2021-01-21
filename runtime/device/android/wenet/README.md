# Wenet Android Runtime

You can use our prebuilt APK or build your APK by source code.

## Prebuilt APK

* [Chinese APK, powered by AIshell data(TODO)]()
* [English APK, powered by LibriSpeech data(TODO)]()

## Build your APK

### Build model

You can use our pretrained model(click the following link to download),

* [AIshell](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210121_unified_transformer_device.tar.gz)
* [LibriSpeech(TODO)]()

Or train your own model by WeNet pipeline with your own data.

### Build APK

When your model is ready, put `final.zip` and `words.txt` into android assets (`app/src/main/assets`) folder,
then just build and run. Here is a gif demo, which shows how our device streaming e2e works,
please note there is no network connection ^\_^.

![Runtime android demo](../../../../docs/images/runtime_android.gif)
