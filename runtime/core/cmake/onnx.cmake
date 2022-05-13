FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.9.0/onnxruntime-linux-x64-1.9.0.tgz
    URL_HASH SHA256=f386ab80e9d6d41f14ed9e61bff4acc6bf375770691bc3ba883ba0ba3cabca7f
)
FetchContent_MakeAvailable(onnxruntime)
include_directories("${onnxruntime_SOURCE_DIR}/include")
link_directories("${onnxruntime_SOURCE_DIR}/lib")
