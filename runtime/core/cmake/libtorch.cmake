if(TORCH)
  if(NOT ANDROID)
    set(PYTORCH_VERSION "1.10.0")
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
      if(${CMAKE_BUILD_TYPE} MATCHES "Release")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=d7043b7d7bdb5463e5027c896ac21b83257c32c533427d4d0d7b251548db8f4b")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-${PYTORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=d98c1b6d425ce62a6d65c16d496ef808fb2e7053d706202c536a7e437a5ade86")
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(CXX11_ABI)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=6d7be1073d1bd76f6563572b2aa5548ad51d5bc241d6895e3181b7dc25554426")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=16961222938b205a6a767b0b0b9f5e3b1f8740aa1f3475580e33cfd5952b1a44")
      endif()
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
    FetchContent_MakeAvailable(libtorch)
    find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")

    if(MSVC)
      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
      file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
    endif()
  else()
    # Change version in runtime/device/android/wenet/app/build.gradle.
    file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
    file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")
    find_library(PYTORCH_LIBRARY pytorch_jni
      PATHS ${PYTORCH_LINK_DIRS}
      NO_CMAKE_FIND_ROOT_PATH
    )
    find_library(FBJNI_LIBRARY fbjni
      PATHS ${PYTORCH_LINK_DIRS}
      NO_CMAKE_FIND_ROOT_PATH
    )
    include_directories(
      ${PYTORCH_INCLUDE_DIRS}
      ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
    )
  endif()
  add_definitions(-DUSE_TORCH)
endif()
