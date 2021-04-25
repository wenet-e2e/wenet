cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(kaldi)

add_library(kaldi
base/kaldi-math.cc
)

target_link_libraries(kaldi PUBLIC utils)
