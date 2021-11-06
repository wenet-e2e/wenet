cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

option(GRPC "whether to build gRPC" OFF)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi  # for not changing c++ header names in kaldi source files
)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()

# third_party: gflags
FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
  URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# third_party: gtest
FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.10.0.zip
  URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
)
FetchContent_MakeAvailable(googletest)

# third_party: boost
FetchContent_Declare(boost
  URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})

# third_party: libtorch use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.10.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=d7043b7d7bdb5463e5027c896ac21b83257c32c533427d4d0d7b251548db8f4b")
    set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=6d7be1073d1bd76f6563572b2aa5548ad51d5bc241d6895e3181b7dc25554426")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip")
    set(URL_HASH "SHA256=07cac2c36c34f13065cb9559ad5270109ecbb468252fb0aeccfd89322322a2b5")
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()
FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
  URL_HASH ${URL_HASH}
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

# utils
add_library(utils STATIC
  utils/string.cc
  utils/utils.cc
)

# The original openfst uses GNU Build System to run configure and build.
# So, we use "OpenFST port for Windows" to build openfst with cmake in Windows.
# Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
# To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
if(NOT MSVC)
  set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
  set(openfst_BINARY_DIR ${fc_base}/openfst-build CACHE PATH "OpenFST build directory")
  set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix CACHE PATH "OpenFST prefix directory")
  ExternalProject_Add(openfst
    URL               https://github.com/mjansche/openfst/archive/1.6.5.zip
    URL_HASH          SHA256=b720357a464f42e181d7e33f60867b54044007f50baedc8f4458a3926f4a5a78
    SOURCE_DIR        ${openfst_SOURCE_DIR}
    BINARY_DIR        ${openfst_BINARY_DIR}
    CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                        "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                        "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                        "LIBS=-lgflags_nothreads -lglog -lpthread"
    COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    BUILD_COMMAND     make -j 4
  )
  add_dependencies(openfst gflags glog)
  add_dependencies(utils openfst)
  link_directories(${openfst_PREFIX_DIR}/lib)
  target_link_libraries(utils PUBLIC fst dl)
else()
  execute_process(COMMAND bootstrap.bat WORKING_DIRECTORY ${boost_SOURCE_DIR})
  execute_process(COMMAND b2.exe WORKING_DIRECTORY ${boost_SOURCE_DIR})
  link_directories(${boost_SOURCE_DIR}/stage/lib)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
  FetchContent_Declare(openfst
    URL      https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.5.1.tar.gz
    URL_HASH SHA256=02c49b559c3976a536876063369efc0e41ab374be1035918036474343877046e
  )
  FetchContent_MakeAvailable(openfst)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR})
  add_dependencies(fst gflags glog)
  target_link_libraries(fst PUBLIC gflags_nothreads_static glog)
  target_link_libraries(utils PUBLIC fst)
endif()
include_directories(${openfst_SOURCE_DIR}/src/include)

# frontend
add_library(frontend STATIC
  frontend/feature_pipeline.cc
  frontend/fft.cc
)
target_link_libraries(frontend PUBLIC utils)

# kaldi: wfst based decoder
add_subdirectory(kaldi)

# decoder
add_library(decoder STATIC
  decoder/context_graph.cc
  decoder/ctc_prefix_beam_search.cc
  decoder/ctc_wfst_beam_search.cc
  decoder/ctc_endpoint.cc
  decoder/torch_asr_decoder.cc
  decoder/torch_asr_model.cc
)
target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES} kaldi-decoder utils post_processor)

add_executable(ctc_prefix_beam_search_test decoder/ctc_prefix_beam_search_test.cc)
target_link_libraries(ctc_prefix_beam_search_test PUBLIC gtest_main gmock decoder)
add_test(CTC_PREFIX_BEAM_SEARCH_TEST ctc_prefix_beam_search_test)

# post_processor
add_library(post_processor STATIC
  post_processor/post_processor.cc
)
target_link_libraries(post_processor PUBLIC utils)
add_executable(post_processor_test post_processor/post_processor_test.cc)
target_link_libraries(post_processor_test PUBLIC post_processor gtest_main gmock)
add_test(POST_PROCESSOR_TEST post_processor_test)

# websocket
add_library(websocket STATIC
  websocket/websocket_client.cc
  websocket/websocket_server.cc
)
target_link_libraries(websocket PUBLIC decoder frontend)

# binary
add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC frontend decoder)
add_executable(websocket_client_main bin/websocket_client_main.cc)
target_link_libraries(websocket_client_main PUBLIC websocket)
add_executable(websocket_server_main bin/websocket_server_main.cc)
target_link_libraries(websocket_server_main PUBLIC websocket)
add_executable(label_checker_main bin/label_checker_main.cc)
target_link_libraries(label_checker_main PUBLIC frontend decoder)

if(GRPC)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grpc)
  # third_party: grpc
  # On how to build grpc, you may refer to https://github.com/grpc/grpc
  # We recommend manually recursive clone the repo to avoid internet connection problem
  FetchContent_Declare(gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.37.1
  )
  FetchContent_MakeAvailable(gRPC)

  # compile wenet.proto
  set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grpc")
  add_custom_command(
    OUTPUT ${PROTO_DIR}/wenet.pb.cc
           ${PROTO_DIR}/wenet.pb.h
           ${PROTO_DIR}/wenet.grpc.pb.cc
           ${PROTO_DIR}/wenet.grpc.pb.h
    COMMAND ${protobuf_BINARY_DIR}/protoc
    ARGS --grpc_out "${PROTO_DIR}"
      --cpp_out "${PROTO_DIR}"
      -I "${PROTO_DIR}"
      --plugin=protoc-gen-grpc=${grpc_BINARY_DIR}/grpc_cpp_plugin
      wenet.proto)

  # grpc_server/client
  link_directories(${protobuf_BINARY_DIR}/lib)
  add_library(wenet_grpc STATIC
    grpc/grpc_client.cc
    grpc/grpc_server.cc
    grpc/wenet.pb.cc
    grpc/wenet.grpc.pb.cc
  )
  target_link_libraries(wenet_grpc PUBLIC grpc++ grpc++_reflection decoder frontend)

  add_executable(grpc_server_main bin/grpc_server_main.cc)
  target_link_libraries(grpc_server_main PUBLIC wenet_grpc)
  add_executable(grpc_client_main bin/grpc_client_main.cc)
  target_link_libraries(grpc_client_main PUBLIC wenet_grpc)
endif(GRPC)

