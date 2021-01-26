cmake_minimum_required(VERSION 3.4.1)
set(TARGET wenet)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR})

# Pytorch
set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")
find_library(PYTORCH_LIBRARY pytorch_jni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)
find_library(FBJNI_LIBRARY fbjni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)
include_directories(${PYTORCH_INCLUDE_DIRS}
    ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include)

# frontend
add_library(frontend STATIC
    frontend/feature_pipeline.cc
    frontend/fft.cc)
target_link_libraries(frontend PUBLIC log)

# decoder
add_library(decoder STATIC
    decoder/ctc_prefix_beam_search.cc
    decoder/torch_asr_decoder.cc
    decoder/torch_asr_model.cc)
target_link_libraries(decoder PUBLIC log ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})

link_libraries(frontend decoder android)

add_library(${TARGET} SHARED wenet.cc)
add_executable(decoder_main bin/decoder_main.cc)
