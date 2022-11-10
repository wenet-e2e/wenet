# WeNet On-device ASR iOS Demo

## Build application from source code

### 1) Generate cmake project and install LibTorch pod

```
cd runtime/ios/build
cmake .. -G Xcode -DTORCH=OFF -DONNX=OFF -DIOS=ON -DGRAPH_TOOLS=ON -DBUILD_TESTING=OFF -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DPLATFORM=OS64 -DENABLE_BITCODE=FALSE
pod install
```

### 2) Remove executable targets in wenet project

Open wenet.xcodeproj in runtime/ios/build folder with Xcode, remove all 6 executable targets, leave static library targets only, close Xcode to save workspace.

### 3) Build static libraries

```
# Build debug version
cmake --build . --config Debug

# Build release version
cmake --build . --config Release
```

### 4) Build and run iOS application

You can use our pretrained model (click the following link to download):

[AISHELL-1](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_libtorch.tar.gz)
| [AISHELL-2](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_libtorch.tar.gz)
| [GigaSpeech](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_libtorch.tar.gz)
| [LibriSpeech](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_libtorch.tar.gz)
| [Multi-CN](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/multi_cn/20210815_unified_conformer_libtorch.tar.gz)


Or you can train your own model using WeNet training pipeline on your data.

When your model is ready, put `final.zip` and `units.txt` into model (`WenetDemo/WenetDemo/model`) folder.

Open WenetDemo.xcodeproj with Xcode, build and run on iOS device.
