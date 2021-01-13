cmake_minimum_required(VERSION 3.4.1)
set(TARGET wenet)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB wenet_SOURCES
        ${CMAKE_SOURCE_DIR}/wenet.cpp
        ${CMAKE_SOURCE_DIR}/glog/logging.h
        )

add_library(${TARGET} SHARED
        ${wenet_SOURCES}
        )

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

target_compile_options(${TARGET} PRIVATE
        -fexceptions
        )

find_library(PYTORCH_LIBRARY pytorch_jni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH
        )

target_include_directories(${TARGET} PRIVATE
        ${PYTORCH_INCLUDE_DIRS}
        ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
        )

target_link_libraries(${TARGET}
        ${PYTORCH_LIBRARY}
        log
        )