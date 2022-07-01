set(decoder_srcs
  asr_decoder.cc
  asr_model.cc
  context_graph.cc
  ctc_prefix_beam_search.cc
  ctc_wfst_beam_search.cc
  ctc_endpoint.cc
)

if(NOT TORCH AND NOT ONNX)
  message(FATAL_ERROR "Please build with TORCH or ONNX!!!")
endif()
if(TORCH)
  list(APPEND decoder_srcs torch_asr_model.cc)
endif()
if(ONNX)
  list(APPEND decoder_srcs onnx_asr_model.cc)
endif()

add_library(decoder STATIC ${decoder_srcs})
target_link_libraries(decoder PUBLIC kaldi-decoder frontend post_processor utils)

if(ANDROID)
  target_link_libraries(decoder PUBLIC ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})
else()
  if(TORCH)
    target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES})
  endif()
  if(ONNX)
    target_link_libraries(decoder PUBLIC onnxruntime)
  endif()
endif()
