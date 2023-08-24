if(BPU)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(EASY_DNN_URL "https://github.com/xingchensong/toolchain_pkg/releases/download/easy_dnn/easy_dnn.0.4.11.tar.gz")
      set(URL_HASH "SHA256=a1a6f77d1baae7181d75ec5d37a2ee529ac4e1c4400babd6ceb1c007392a4904")
    else()
      message(FATAL_ERROR "Unsupported CMake System Processor '${CMAKE_SYSTEM_PROCESSOR}' (expected 'aarch64')")
    endif()
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
  endif()

  FetchContent_Declare(easy_dnn
    URL ${EASY_DNN_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(easy_dnn)
  include_directories(${easy_dnn_SOURCE_DIR}/easy_dnn/0.4.11_linux_aarch64-j3_hobot_gcc6.5.0/files/easy_dnn/include)
  include_directories(${easy_dnn_SOURCE_DIR}/dnn/1.7.0_linux_aarch64-j3_hobot_gcc6.5.0/files/dnn/include)
  include_directories(${easy_dnn_SOURCE_DIR}/hlog/0.4.7_linux_aarch64-j3_hobot_gcc6.5.0/files/hlog/include)
  link_directories(${easy_dnn_SOURCE_DIR}/easy_dnn/0.4.11_linux_aarch64-j3_hobot_gcc6.5.0/files/easy_dnn/lib)
  link_directories(${easy_dnn_SOURCE_DIR}/dnn/1.7.0_linux_aarch64-j3_hobot_gcc6.5.0/files/dnn/lib)
  link_directories(${easy_dnn_SOURCE_DIR}/hlog/0.4.7_linux_aarch64-j3_hobot_gcc6.5.0/files/hlog/lib)

  add_definitions(-DUSE_BPU)
  # NOTE(xcsong): Reasons for adding flag `-fuse-ld=gold`:
  #   https://stackoverflow.com/questions/59915966/unknown-gcc-linker-error-but-builds-sucessfully/59916438#59916438
  #   https://github.com/tensorflow/tensorflow/issues/47849
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()
