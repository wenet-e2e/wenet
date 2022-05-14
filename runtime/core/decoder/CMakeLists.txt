set(decoder_srcs
  asr_decoder.cc
  asr_model.cc
  context_graph.cc
  ctc_prefix_beam_search.cc
  ctc_wfst_beam_search.cc
  ctc_endpoint.cc
  torch_asr_model.cc
)
if(ONNX)
  list(APPEND decoder_srcs onnx_asr_model.cc)
endif()

add_library(decoder STATIC ${decoder_srcs})

target_link_libraries(decoder PUBLIC kaldi-decoder post_processor utils)

if(ANDROID)
  target_link_libraries(decoder PUBLIC ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})
else()
  target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES})
  if(ONNX)
    target_link_libraries(decoder PUBLIC onnxruntime)
  endif()
endif()
