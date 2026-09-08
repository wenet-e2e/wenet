FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.7.1.zip
  URL_HASH SHA256=c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
