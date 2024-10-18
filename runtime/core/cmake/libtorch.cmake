if(TORCH)
  set(TORCH_VERSION "2.2.0")
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
        set(URL_HASH "SHA256=96bc833184a7c13a088a2a83cab5a2be853c0c9d9f972740a50580173d0c796d")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-${TORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=5b7dbabbecd86051b800ce0a244f15b89e9de0f8b5370e5fa65668aa37ecb878")
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(CXX11_ABI)
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=62cd3001a2886d2db125aabc3be5c4fb66b3e34b32727d84323968f507ee8e32")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=a2b0f51ff59ef2787a82c36bba67f7380236a6384dbbd2459c558989af27184f")
        endif()
      else()
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=e1f6bc48403022ff4680c7299cc8b160df146892c414b8a6b6f7d5aff65bcbce")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=f9c887085207f9500357cae4324a53c3010b8890397db915d7dbefb9183c7964")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${TORCH_VERSION}.zip")
        set(URL_HASH "SHA256=a2ac530e5db2f5be33fe7f7e3049b9a525ee60b110dbb1e08835e22002756ed1")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${TORCH_VERSION}.zip")
        set(URL_HASH "SHA256=300940c6b1d4402ece72d31cd5694d9579dcfb23b7cf6b05676006411f9b516c")
      endif()
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
