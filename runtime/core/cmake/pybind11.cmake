FetchContent_Declare(pybind11
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.zip
  URL_HASH SHA256=d1646e6f70d8a3acb2ddd85ce1ed543b5dd579c68b8fb8e9638282af20edead8
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(${pybind11_SOURCE_DIR})