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
        set(URL_HASH "SHA256=77815aa799f15e91b6fbb0216ac78cc0479adb5cd0ca662072241484cf23f667")
      else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-${TORCH_VERSION}%2Bcpu.zip")
        set(URL_HASH "SHA256=5f887c02d9abf805c8b53fef89bf5a4dab9dd78771754344e73c98d9c484aa9d")
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      if(CXX11_ABI)
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=04f699d5181048b0062ef52de1df44b46859b8fbeeee12abdbcb9aac63e2a14b")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=7796249faa9828a53b72d3f616fc97a1d9e87e6a35ac72b392ca1ddc7b125188")
        endif()
      else()
        if(NOT GPU)
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip")
          set(URL_HASH "SHA256=0e86d364d05b83c6c66c3bb32e7eee932847843e4085487eefd9b3bbde4e2c58")
        else()
          set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip")
          set(URL_HASH "SHA256=f70cfae25b02ff419e1d51ad137a746941773d2c4b0155a44b4b6b50702d661a")
        endif()
      endif()
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH_VERSION}.zip")
      set(URL_HASH "SHA256=ce744d2d27a96df8f34d4227e8b1179dad5a76612dc7230b61db65affce6e7bd")
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
