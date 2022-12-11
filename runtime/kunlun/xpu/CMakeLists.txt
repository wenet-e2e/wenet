message("cmake build type is ${CMAKE_BUILD_TYPE} .")

if(XPU)
  list(APPEND xpu_conformer_srcs ./xpu_asr_model.cc)
  list(APPEND xpu_conformer_srcs ./xpu_conformer.cpp)
  list(APPEND xpu_conformer_srcs ./xpu_util.cpp)
  message(STATUS "Use src_files: [ ${xpu_conformer_srcs} ] to compile xpu_conformer.a .")

  # compile xpu_conformer.a
  add_library(xpu_conformer STATIC ${xpu_conformer_srcs})
  target_link_libraries(xpu_conformer PUBLIC xpuapi xpurt)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread -lrt -lm -ldl")

set(SRC_FILES ./conformer_test.cpp ./xpu_conformer.cpp ./xpu_util.cpp)
message(STATUS "Use src_files: [ ${SRC_FILES} ] to compile xpu_conformer_test.")

add_executable(xpu_conformer_test ${SRC_FILES})
target_link_libraries(xpu_conformer_test -lxpuapi -lxpurt)
