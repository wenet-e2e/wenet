cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})
option(GRPC "whether to build gRPC" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

include(libtorch)
include(openfst)
include(pybind11)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi
)

# utils
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)
if(NOT MSVC)
  add_dependencies(utils openfst)
  target_link_libraries(utils PUBLIC fst dl)
endif()

# frontend
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/frontend)
add_subdirectory(frontend)

# post_processor
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/post_processor)
add_subdirectory(post_processor)

# kaldi: wfst based decoder
add_subdirectory(kaldi)

# decoder
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/decoder)
add_subdirectory(decoder)
target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES} post_processor)

add_library(wenet_api SHARED api/wenet_api.cc)
target_link_libraries(wenet_api PUBLIC decoder frontend)

# wenet api
pybind11_add_module(_wenet cpp/binding.cc)
target_link_libraries(_wenet PRIVATE wenet_api)
