project(wenet_binding)

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(NOT WENET_RUNTIME_CORE_DIR)
  set(WENET_RUNTIME_CORE_DIR ${CMAKE_SOURCE_DIR}/../server/x86)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(LIBRARY_OUTPUT_DIRECTORY, ${CMAKE_BINARY_DIR}/bin)
set(ARCHIVE_OUTPUT_DIRECTORY, ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(${WENET_RUNTIME_CORE_DIR})
if(NOT MSVC)
  link_directories(${openfst_PREFIX_DIR}/lib)
else()
  link_directories(${boost_SOURCE_DIR}/stage/lib)
endif()

# TODO: use boost from wenet runtime
FetchContent_Declare(boost
  URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
  )
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})
link_directories(${boost_SOURCE_DIR}/stage/lib)

find_package(Torch REQUIRED)
message("wenet_runtime_binding CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_SOURCE_DIR}/src/include)

add_library(runtime_wrapper SHARED
  runtime_wrapper.cc
  )
target_include_directories(runtime_wrapper PUBLIC
  ${WENET_RUNTIME_CORE_DIR}
  ${WENET_RUNTIME_CORE_DIR}/kaldi
  ${openfst_SOURCE_DIR}/src/include
  ${gflags_BINARY_DIR}/include
  ${glog_SOURCE_DIR}/src ${glog_BINARY_DIR}
  )
target_link_libraries(runtime_wrapper
  frontend
  decoder
  )

add_subdirectory(python/pybind11)
pybind11_add_module(_pywrap_wenet MODULE
  python/py_asr_wrapper.cc
  )

target_link_libraries(_pywrap_wenet PRIVATE
  runtime_wrapper
  )

# for c so
add_library(cruntime SHARED
  wenet.cc
  )
target_link_libraries(cruntime
  runtime_wrapper
  )

# hard code
function(mv_wrap_lib)
  set(libtorch_SOURCE_DIR "${WENET_RUNTIME_CORE_DIR}/fc_base/libtorch-src/")
  set(openfst_SOURCE_DIR "${WENET_RUNTIME_CORE_DIR}/fc_base/openfst-build/")
  set(python_LIB_DIR "${CMAKE_SOURCE_DIR}/python/wenet/lib/")

  if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_custom_command(TARGET _pywrap_wenet  POST_BUILD
      COMMAND rm -rf  ${python_LIB_DIR}/*.so
      COMMAND cp _pywrap_wenet*.so ${python_LIB_DIR}
      COMMAND cp libruntime_wrapper*.so ${python_LIB_DIR}

      COMMAND cp -L ${libtorch_SOURCE_DIR}/lib/libtorch_cpu.so ${python_LIB_DIR}
      COMMAND cp -L ${libtorch_SOURCE_DIR}/lib/libtorch.so ${python_LIB_DIR}
      COMMAND cp -L ${libtorch_SOURCE_DIR}/lib/libc10.so ${python_LIB_DIR}
      COMMAND cp -L ${libtorch_SOURCE_DIR}/lib/libgomp-*.so.1 ${python_LIB_DIR}
      COMMAND cp  ${openfst_SOURCE_DIR}/src/lib/.libs/libfst.so* ${python_LIB_DIR}
      # python build wheel
      COMMAND cd ${CMAKE_SOURCE_DIR}/python/ &&  ${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel  --python-tag  ${PYTHON_TAG} --plat-name linux-x86_64
      )
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
  endif()

endfunction()
mv_wrap_lib()
