cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
option(FST_HAVE_BIN "whether to build fst binaries" OFF)
option(BUILD_TESTING "whether to build unit test" ON)
option(GRPC "whether to build with gRPC" OFF)
option(TORCH "whether to build with Torch" ON)
option(ONNX "whether to build with ONNX" OFF)

set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
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

include(boost)
include(libtorch)
include(onnx)
include(openfst)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)

add_subdirectory(utils)
if(NOT MSVC)
  add_dependencies(utils openfst)
endif()
add_subdirectory(frontend)
add_subdirectory(post_processor)
add_subdirectory(kaldi)  # kaldi: wfst based decoder
add_subdirectory(decoder)
add_subdirectory(api)

# binary
add_executable(api_main bin/api_main.cc)
target_link_libraries(api_main PUBLIC wenet_api)

add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC decoder frontend)

add_executable(websocket_client_main
  bin/websocket_client_main.cc
  websocket/websocket_client.cc
)
target_link_libraries(websocket_client_main PUBLIC frontend)

add_executable(websocket_server_main
  bin/websocket_server_main.cc
  websocket/websocket_server.cc
)
target_link_libraries(websocket_server_main PUBLIC decoder frontend)

add_executable(label_checker_main bin/label_checker_main.cc)
target_link_libraries(label_checker_main PUBLIC decoder frontend)

if(BUILD_TESTING)
  include(gtest)
  add_subdirectory(test)
endif()

if(GRPC)
  include(grpc)
  add_subdirectory(grpc)

  add_executable(grpc_server_main bin/grpc_server_main.cc)
  target_link_libraries(grpc_server_main PUBLIC wenet_grpc)

  add_executable(grpc_client_main bin/grpc_client_main.cc)
  target_link_libraries(grpc_client_main PUBLIC wenet_grpc)
endif()
