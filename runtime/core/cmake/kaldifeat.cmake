# Copyright     2022 veelion (veelion@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(GPU)
  set(kaldifeat_URL "https://github.com/csukuangfj/kaldifeat/archive/refs/tags/v1.21.zip")
  set(kaldifeat_HASH "SHA256=10652d930dee12d71d04da3f5b3b1bd618fa2f1af6723eb0e70d7267bfa57fe1")
  set(kaldifeat_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(kaldifeat_BUILD_PYMODULE OFF CACHE BOOL "" FORCE)
  set(PYTHON_EXECUTABLE "python")
  list(REMOVE_AT CMAKE_MODULE_PATH 0)  # hide wenet's cmake/xx.cmake from kaldifeat's

  FetchContent_Declare(kaldifeat
      URL               ${kaldifeat_URL}
      URL_HASH          ${kaldifeat_HASH}
  )
  FetchContent_MakeAvailable(kaldifeat)
  include_directories(
      ${kaldifeat_SOURCE_DIR}
  )
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)  # use wenet's cmake/xx.cmake
endif()
