cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
option(GRAPH_TOOLS "whether to build TLG graph tools" OFF)
option(BUILD_TESTING "whether to build unit test" ON)

option(GRPC "whether to build with gRPC" OFF)
# TODO(Binbin Zhang): Change websocket to OFF since it depends on boost
#                     which is a very big library
option(WEBSOCKET "whether to build with websocket" OFF)
option(BPU "whether to build with BPU" ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Keep the same with openfst, -fPIC or -fpic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")

# Include all dependency
include(openfst)
# This CMakeLists.txt is only used for horizon bpu, so remove the contents
#  about onnx, libtorch, gpu and windows.
include(bpu)
# Compile bpu_asr_model
add_subdirectory(bpu)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)

include(wetextprocessing)

# Build all libraries
add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(post_processor)
add_subdirectory(kaldi)  # kaldi: wfst based decoder
add_subdirectory(decoder)
add_subdirectory(api)

# Optionally, you can build with websocket
if(WEBSOCKET)
  include(boost)
  add_subdirectory(websocket)
endif()

# Optionally, you can build with gRPC
if(GRPC)
  include(grpc)
  add_subdirectory(grpc)
endif()

# Build all bins
add_subdirectory(bin)

# Unit Test
if(BUILD_TESTING)
  include(gtest)
  add_subdirectory(test)
endif()
