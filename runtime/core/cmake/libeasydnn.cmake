if(BPU)
  set(EASYDNN_VERSION "0.4.11")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(LIBEASYDNN_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
      set(URL_HASH "SHA256=5820d9f343df73c63b6b2b174a1ff62575032e171c9564bcf92060f46827d0ac")
    else()
      message(FATAL_ERROR "Unsupported CMake System Processor '${CMAKE_SYSTEM_PROCESSOR}' (expected 'aarch64')")
    endif()
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
  endif()

  FetchContent_Declare(libeasydnn
    URL ${LIBEASYDNN_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(libeasydnn)
  include_directories(${libeasydnn_SOURCE_DIR}/include)
  link_directories(${libeasydnn_SOURCE_DIR}/lib)

  add_definitions(-DUSE_BPU)
endif()
