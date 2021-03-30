cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

include_directories(${CMAKE_SOURCE_DIR})

include(FetchContent)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

# third_party: gflags
FetchContent_Declare(
  gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.0.zip
  URL_HASH SHA256=99f9e8b63ea53e4b23f0fd0f68a6f1e397d5512be36716c17cc75966a90f0d57
)

# third_party: glog
FetchContent_Declare(
  glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)

# third_party: gtest
FetchContent_Declare(
  googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)

# third_party: boost
FetchContent_Declare(
  boost
  URL      https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)

# third_party: libtorch 1.6.0, use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.6.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=dacb2af654f1cffa43b0f3187ce945e1c93305b4c8bb60361c7a8cb543e1da82")
    set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=c6c0d3a87039338f7812a1ae343b9e48198536f20d1415b0e5a9a15ba7b90b3f")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip")
    set(URL_HASH "SHA256=e1140bed7bd56c26638bae28aebbdf68e588a9fae92c6684645bcdd996e4183c")
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()
FetchContent_Declare(
  libtorch
  URL      ${LIBTORCH_URL}
  URL_HASH ${URL_HASH}
)

set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(gflags glog googletest boost libtorch)
include_directories(${boost_SOURCE_DIR})
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
if(MSVC)
  execute_process(COMMAND bootstrap.bat WORKING_DIRECTORY ${boost_SOURCE_DIR})
  execute_process(COMMAND b2.exe WORKING_DIRECTORY ${boost_SOURCE_DIR})
  link_directories(${boost_SOURCE_DIR}/stage/lib)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
endif(MSVC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
endif()

# utils
add_library(utils STATIC utils/utils.cc)
target_link_libraries(utils PUBLIC glog)

# frontend
add_library(frontend STATIC
  frontend/feature_pipeline.cc
  frontend/fft.cc
)
target_link_libraries(frontend PUBLIC glog)

# decoder
add_library(decoder STATIC
  decoder/ctc_prefix_beam_search.cc
  decoder/torch_asr_decoder.cc
  decoder/torch_asr_model.cc
  utils/utils.cc
)
target_link_libraries(decoder PUBLIC glog ${TORCH_LIBRARIES})

add_executable(ctc_prefix_beam_search_test decoder/ctc_prefix_beam_search_test.cc)
target_link_libraries(ctc_prefix_beam_search_test PUBLIC gtest_main decoder gmock)
add_test(CTC_PREFIX_BEAM_SEARCH_TEST ctc_prefix_beam_search_test)

# websocket
add_library(websocket STATIC
  websocket/websocket_client.cc
  websocket/websocket_server.cc
)
target_link_libraries(websocket PUBLIC glog frontend decoder)

# binary
add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC gflags frontend decoder utils)
add_executable(websocket_client_main bin/websocket_client_main.cc)
target_link_libraries(websocket_client_main PUBLIC gflags websocket)
add_executable(websocket_server_main bin/websocket_server_main.cc)
target_link_libraries(websocket_server_main PUBLIC gflags websocket)
