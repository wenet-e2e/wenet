cmake_minimum_required(VERSION 3.4.1)
set(TARGET wenet)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)

option(TORCH "whether to build with Torch" ON)
option(ONNX "whether to build with ONNX" OFF)
set(CMAKE_VERBOSE_MAKEFILE on)
set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
string(REPLACE "-Wl,--exclude-libs,libgcc_real.a" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

include(libtorch)
include(openfst)

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/kaldi
)

add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(post_processor)
add_subdirectory(kaldi)  # kaldi: wfst based decoder
add_subdirectory(decoder)

link_libraries(frontend decoder android)
add_library(${TARGET} SHARED wenet.cc)

add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC libc++_shared.so)
