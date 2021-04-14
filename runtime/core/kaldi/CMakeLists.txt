cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(kaldi)

include_directories(${CMAKE_SOURCE_DIR}/kaldi)

add_library(kaldi
decoder/lattice-faster-decoder.cc
base/kaldi-math.cc
)

target_link_libraries(kaldi PUBLIC utils)
