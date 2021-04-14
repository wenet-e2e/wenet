cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(kaldi)

include_directories(${CMAKE_SOURCE_DIR}/kaldi)

add_library(kaldi
base/kaldi-math.cc
decoder/lattice-faster-decoder.cc
decoder/lattice-faster-online-decoder.cc
)

target_link_libraries(kaldi PUBLIC utils)
