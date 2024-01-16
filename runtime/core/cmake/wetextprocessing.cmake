if(NOT ANDROID)
  FetchContent_Declare(wetextprocessing
    GIT_REPOSITORY https://github.com/wenet-e2e/WeTextProcessing.git
    GIT_TAG        origin/master
  )
  FetchContent_MakeAvailable(wetextprocessing)
  include_directories(${wetextprocessing_SOURCE_DIR}/runtime )
  add_subdirectory(${wetextprocessing_SOURCE_DIR}/runtime/utils)
  add_subdirectory(${wetextprocessing_SOURCE_DIR}/runtime/processor)
else()
  include(ExternalProject)
  set(ANDROID_CMAKE_ARGS
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DANDROID_ABI=${ANDROID_ABI}
    -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
    -DCMAKE_CXX_FLAGS=-I${openfst_BINARY_DIR}/include
    -DCMAKE_EXE_LINKER_FLAGS=-L${openfst_BINARY_DIR}/${ANDROID_ABI}
  )
  ExternalProject_Add(wetextprocessing
    URL https://github.com/wenet-e2e/WeTextProcessing/archive/refs/tags/0.1.3.tar.gz
    URL_HASH SHA256=2f1c81649b2f725a5825345356be9dccb9699965cf44c9f0e842f5c0d4b6ba61
    SOURCE_SUBDIR runtime
    CMAKE_ARGS ${ANDROID_CMAKE_ARGS}
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(wetextprocessing SOURCE_DIR BINARY_DIR)
  include_directories(${SOURCE_DIR}/runtime)
  link_directories(${BINARY_DIR}/processor ${BINARY_DIR}/utils)
  link_libraries(wetext_utils)
endif()

