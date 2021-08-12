cmake_minimum_required(VERSION 3.4.1)
set(TARGET wenet)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/kaldi  # for not changing c++ header names in kaldi source files
)
set(CMAKE_VERBOSE_MAKEFILE on)
set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
string(REPLACE "-Wl,--exclude-libs,libgcc_real.a" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(openfst_BINARY_DIR ${build_DIR}/wenet-openfst-android-1.0.1.aar/jni)
include_directories(${openfst_BINARY_DIR}/include)
link_directories(${openfst_BINARY_DIR}/${ANDROID_ABI})
link_libraries(log gflags_nothreads glog fst)

# Pytorch
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

# utils
add_library(utils STATIC
  utils/string.cc
  utils/utils.cc
)

# frontend
add_library(frontend STATIC
  frontend/feature_pipeline.cc
  frontend/fft.cc
)
target_link_libraries(frontend PUBLIC utils)

# kaldi: wfst based decoder
add_subdirectory(kaldi)

# decoder
add_library(decoder STATIC
  decoder/ctc_endpoint.cc
  decoder/ctc_prefix_beam_search.cc
  decoder/ctc_wfst_beam_search.cc
  decoder/torch_asr_decoder.cc
  decoder/torch_asr_model.cc
)
target_link_libraries(decoder PUBLIC kaldi-decoder utils ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})

link_libraries(utils frontend decoder android)
add_library(${TARGET} SHARED wenet.cc)

add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC libc++_shared.so)
