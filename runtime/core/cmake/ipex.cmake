add_definitions(-DUSE_TORCH)
add_definitions(-DUSE_IPEX)
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(FATAL_ERROR "Intel Extension For PyTorch supports only Linux for now")
endif()

if(CXX11_ABI)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip")
  set(URL_HASH "SHA256=137a842d1cf1e9196b419390133a1623ef92f8f84dc7a072f95ada684f394afd")
else()
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip")
  set(URL_HASH "SHA256=90d50350fd24ce5cf9dfbf47888d0cfd9f943eb677f481b86fe1b8e90f7fda5d")
endif()
FetchContent_Declare(libtorch
URL      ${LIBTORCH_URL}
URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

if(CXX11_ABI)
  set(LIBIPEX_URL "https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-2.0.100%2Bcpu.run")
  set(URL_HASH "SHA256=f172d9ebc2ca0c39cc93bb395721194f79767e1bc3f82b13e1edc07d1530a600")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-cxx11-abi-2.0.100%2Bcpu.run")
else()
  set(LIBIPEX_URL "https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-2.0.100%2Bcpu.run")
  set(URL_HASH "SHA256=8392f965dd9b8f6c0712acbb805c7e560e4965a0ade279b47a5f5a8363888268")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-2.0.100%2Bcpu.run")
endif()
FetchContent_Declare(intel_ext_pt
URL                  ${LIBIPEX_URL}
URL_HASH             ${URL_HASH}
DOWNLOAD_DIR         ${FETCHCONTENT_BASE_DIR}
DOWNLOAD_NO_EXTRACT  TRUE
PATCH_COMMAND        bash ${FETCHCONTENT_BASE_DIR}/${LIBIPEX_SCRIPT_NAME} install ${libtorch_SOURCE_DIR}
)
FetchContent_MakeAvailable(intel_ext_pt)
find_package(IPEX REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
