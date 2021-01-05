cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet VERSION 0.1)

include_directories(${CMAKE_SOURCE_DIR})

include(FetchContent)

# gflags
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG        v2.2.0
)
FetchContent_MakeAvailable(gflags)

# glog
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG        v0.4.0
)
FetchContent_MakeAvailable(glog)

# gtest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

# libtorch 1.6.0, use FetchContent_Declare to download, and find_package to find
# since libtorch is not a standard cmake project
FetchContent_Declare(
  libtorch
  URL      https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
  URL_HASH SHA256=c6c0d3a87039338f7812a1ae343b9e48198536f20d1415b0e5a9a15ba7b90b3f
)
FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  FetchContent_Populate(libtorch)
  message(STATUS "libtorch_SOURCE_DIR ${libtorch_SOURCE_DIR}")
  set(Torch_DIR ${libtorch_SOURCE_DIR})
endif()
find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES ${TORCH_LIBRARIES}")

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
)
target_link_libraries(decoder PUBLIC glog "${TORCH_LIBRARIES}")

add_executable(ctc_prefix_beam_search_test decoder/ctc_prefix_beam_search_test.cc)
target_link_libraries(ctc_prefix_beam_search_test PUBLIC glog gtest_main decoder)
add_test(CTC_PREFIX_BEAM_SEARCH_TEST ctc_prefix_beam_search_test)

# binary
add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC gflags frontend decoder)
