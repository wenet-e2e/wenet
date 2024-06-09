add_definitions(-DUSE_TORCH)
add_definitions(-DUSE_IPEX)
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(FATAL_ERROR "Intel Extension For PyTorch supports only Linux for now")
endif()

set(TORCH_VERSION "2.3.0")
set(IPEX_VERSION "2.3.0")

if(CXX11_ABI)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
  set(URL_HASH "SHA256=f60009d2a74b6c8bdb174e398c70d217b7d12a4d3d358cd1db0690b32f6e193b")
else()
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
  set(URL_HASH "SHA256=6b78aff4e586991bb2e040c02b2cfd73bc740059b9d12bcc1c1d7b3c86d2ab88")
endif()
FetchContent_Declare(libtorch
URL      ${LIBTORCH_URL}
URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

if(CXX11_ABI)
  set(LIBIPEX_URL "https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-${IPEX_VERSION}%2Bcpu.run")
  set(URL_HASH "SHA256=8aa3c7c37f5cc2cba450947ca04f565fccb86c3bb98f592142375cfb9016f0d6")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-cxx11-abi-${IPEX_VERSION}%2Bcpu.run")
else()
  set(LIBIPEX_URL "https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-${IPEX_VERSION}%2Bcpu.run")
  set(URL_HASH "SHA256=fecb6244a6cd38ca2d73a45272a6ad8527d1ec2caca512d919daa80adb621814")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-${IPEX_VERSION}%2Bcpu.run")
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
