cmake_minimum_required(VERSION 3.4.1)
set(TARGET wenet)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)

include_directories(${CMAKE_SOURCE_DIR})

# Pytorch
set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")
find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)
find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)
include_directories(
  ${PYTORCH_INCLUDE_DIRS}
  ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
)

set(ANDROID_CMAKE_ARGS
  -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
  -DANDROID_NDK=${ANDROID_NDK}
  -DANDROID_ARM_NEON=ON
  -DANDROID_ABI=${ANDROID_ABI}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
  -DANDROID_PLATFORM=${ANDROID_PLATFORM}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  -DBUILD_TESTING=OFF
  -DBUILD_SHARED_LIBS=ON
)

# third party: gflags
set(GFLAGS_SRC_DIR ${build_DIR}/gflags-src)
set(GFLAGS_BUILD_DIR ${CMAKE_BINARY_DIR}/gflags-build)
set(GFLAGS_LIB_PATH ${GFLAGS_BUILD_DIR}/lib/libgflags.so)
ExternalProject_Add(gflags
  URL              https://github.com/gflags/gflags/archive/v2.2.0.zip
  URL_HASH         SHA256=99f9e8b63ea53e4b23f0fd0f68a6f1e397d5512be36716c17cc75966a90f0d57
  SOURCE_DIR       ${GFLAGS_SRC_DIR}
  BINARY_DIR       ${GFLAGS_BUILD_DIR}
  BUILD_BYPRODUCTS ${GFLAGS_LIB_PATH}
  CMAKE_ARGS       ${ANDROID_CMAKE_ARGS}
  INSTALL_COMMAND  ""
)
include_directories(${GFLAGS_BUILD_DIR}/include)

# utils
add_library(utils STATIC utils/utils.cc)

# frontend
add_library(frontend STATIC
    frontend/feature_pipeline.cc
    frontend/fft.cc)
target_link_libraries(frontend PUBLIC log)

# decoder
add_library(decoder STATIC
    utils/utils.cc
    decoder/ctc_prefix_beam_search.cc
    decoder/torch_asr_decoder.cc
    decoder/torch_asr_model.cc)
target_link_libraries(decoder PUBLIC log ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})
link_libraries(frontend decoder android)

add_library(${TARGET} SHARED wenet.cc)

add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main ${GFLAGS_LIB_PATH} utils libc++_shared.so)
add_dependencies(decoder_main gflags)
