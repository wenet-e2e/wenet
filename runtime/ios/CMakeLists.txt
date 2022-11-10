cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
option(GRAPH_TOOLS "whether to build TLG graph tools" OFF)
option(BUILD_TESTING "whether to build unit test" ON)

option(GRPC "whether to build with gRPC" OFF)
# TODO(Binbin Zhang): Change websocket to OFF since it depends on boost
#                     which is a very big library
option(WEBSOCKET "whether to build with websocket" ON)
option(TORCH "whether to build with Torch" ON)
option(ONNX "whether to build with ONNX" OFF)
option(GPU "whether to build with GPU" OFF)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

# Include all dependency
if(TORCH)
  include(libtorch)
endif()
if(ONNX)
  include(onnx)
endif()
include(openfst)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi
  ${CMAKE_CURRENT_SOURCE_DIR}/build/Pods/LibTorch/install/include
)

# Build all libraries
add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(post_processor)
add_subdirectory(kaldi)  # kaldi: wfst based decoder
add_subdirectory(decoder)

# Unit Test
if(BUILD_TESTING)
  include(gtest)
  add_subdirectory(test)
endif()
