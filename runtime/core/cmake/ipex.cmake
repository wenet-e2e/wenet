add_definitions(-DUSE_TORCH)
add_definitions(-DUSE_IPEX)
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(FATAL_ERROR "Intel Extension For PyTorch supports only Linux for now")
endif()

if(CXX11_ABI)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip")
  set(URL_HASH "SHA256=d52f63577a07adb0bfd6d77c90f7da21896e94f71eb7dcd55ed7835ccb3b2b59")
else()
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip")
  set(URL_HASH "SHA256=bee1b7be308792aa60fc95a4f5274d9658cb7248002d0e333d49eb81ec88430c")
endif()
FetchContent_Declare(libtorch
URL      ${LIBTORCH_URL}
URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

if(CXX11_ABI)
  set(LIBIPEX_URL "http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.13.100%2Bcpu.run")
  set(URL_HASH "SHA256=e26778d68a5b76e5d4e540812433ade4f3e917bafc7a252a1fc1c0d1e0ed9c73")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-cxx11-abi-1.13.100%2Bcpu.run")
else()
  set(LIBIPEX_URL "http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.13.100%2Bcpu.run")
  set(URL_HASH "SHA256=0e0e09384106a1f00f4dbd9b543ff42643b769f654c693ff9ecdaba04c607987")
  set(LIBIPEX_SCRIPT_NAME "libintel-ext-pt-1.13.100%2Bcpu.run")
endif()
FetchContent_Declare(intel_ext_pt
URL                  ${LIBIPEX_URL}
URL_HASH             ${URL_HASH}
DOWNLOAD_DIR         ${FETCHCONTENT_BASE_DIR}
DOWNLOAD_NO_EXTRACT  TRUE
PATCH_COMMAND        bash ${FETCHCONTENT_BASE_DIR}/${LIBIPEX_SCRIPT_NAME} install ${libtorch_SOURCE_DIR}
)
FetchContent_MakeAvailable(intel_ext_pt)
find_package(intel_ext_pt_cpu REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
