message("cmake build type is ${CMAKE_BUILD_TYPE} .")

if(BPU)
  list(APPEND bpu_asr_model_srcs ./bpu_asr_model.cc)
  message(STATUS "Use src_files: [ ${bpu_asr_model_srcs} ] to compile bpu_asr_model .")

  # compile bpu_asr_model
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
  add_library(bpu_asr_model STATIC ${bpu_asr_model_srcs})
  target_link_libraries(bpu_asr_model PUBLIC easy_dnn dnn)
endif()
