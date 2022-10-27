# iOS porting of WeNet

* Step 1. Generate cmake project

```
cd runtime/ios
mkdir build
cd build
cmake .. -G Xcode -DTORCH=OFF -DONNX=OFF -DIOS=ON -DGRAPH_TOOLS=ON -DBUILD_TESTING=OFF -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DPLATFORM=OS64 -DENABLE_BITCODE=FALSE
```

* Step 2. Remove executable targets in wenet project

Open wenet.xcodeproj with Xcode, remove all executable targets, leave static library targets only.

* Step 3. Build static libraries

```
cmake --build . --config Debug
```

* Step 4. Build and run iOS application

Open WenetDemo.xcodeproj, build and run on iOS device.

TODO: Debug application, verify decoding results.