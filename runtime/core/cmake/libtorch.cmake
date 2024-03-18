if(TORCH)
  set(TORCH_VERSION "2.1.0")
  add_definitions(-DUSE_TORCH)
  if(NOT ANDROID)
    if(GPU)
      if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        message(FATAL_ERROR "GPU is supported only Linux, you can use CPU version")
      else()
        add_definitions(-DUSE_GPU)
      endif()
    endif()

    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
      if(${CMAKE_BUILD_TYPE} MATCHES "Release")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=bece54d36377990257e9d028c687c5b6759c5cfec0a0153da83cf6f0f71f648f")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-${TORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=3cc7ba3c3865d86f03d78c2f0878fdbed8b764359476397a5c95cf3bba0d665a")
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(CXX11_ABI)
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=d52f63577a07adb0bfd6d77c90f7da21896e94f71eb7dcd55ed7835ccb3b2b59")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=80f089939de20e68e3fcad4dfa72a26c8bf91b5e77b11042f671f39ebac35865")
        endif()
      else()
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=0e86d364d05b83c6c66c3bb32e7eee932847843e4085487eefd9b3bbde4e2c58")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=90159ecce3ff451f3ef3f657493b6c7c96759c3b74bbd70c1695f2ea2f81e1ad")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH_VERSION}.zip")
      set(URL_HASH "SHA256=a8f80050b95489b4e002547910410c2c230e9f590ffab2482e19e809afe4f7aa")
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
      add_definitions(-DIOS)
    else()
      message(FATAL_ERROR "Unsupported System '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux', 'Darwin' or 'iOS')")
    endif()

    # iOS use LibTorch from pod install
    if(NOT IOS)
      FetchContent_Declare(libtorch
        URL      ${LIBTORCH_URL}
        URL_HASH ${URL_HASH}
      )
      FetchContent_MakeAvailable(libtorch)
      find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
    endif()

    if(MSVC)
      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
      file(COPY ${TORCH_DLLS} DESTINATION ${CMAKE_BINARY_DIR})
    endif()
  else()
    # Change version in runtime/android/app/build.gradle.
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
endif()
