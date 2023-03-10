set(decoder_srcs
  asr_decoder.cc
  asr_model.cc
  context_graph.cc
  ctc_prefix_beam_search.cc
  ctc_wfst_beam_search.cc
  ctc_endpoint.cc
)

if(NOT TORCH AND NOT ONNX AND NOT XPU AND NOT IOS AND NOT BPU AND NOT OPENVINO)
  message(FATAL_ERROR "Please build with TORCH or ONNX or OPENVINO or XPU or IOS or BPU!!!")
endif()
if(TORCH OR IOS)
  list(APPEND decoder_srcs torch_asr_model.cc)
endif()
if(ONNX)
  list(APPEND decoder_srcs onnx_asr_model.cc)
endif()

add_library(decoder STATIC ${decoder_srcs})
target_link_libraries(decoder PUBLIC kaldi-decoder frontend
                      post_processor utils)

if(ANDROID)
  target_link_libraries(decoder PUBLIC ${PYTORCH_LIBRARY} ${FBJNI_LIBRARY})
else()
  if(TORCH)
    target_link_libraries(decoder PUBLIC ${TORCH_LIBRARIES})
  endif()
  if(ONNX)
    target_link_libraries(decoder PUBLIC onnxruntime)
  endif()
  if(BPU)
    target_link_libraries(decoder PUBLIC bpu_asr_model)
  endif()
  if(XPU)
    target_link_libraries(decoder PUBLIC xpu_conformer)
  endif()
  if(OPENVINO)
    target_link_libraries(decoder PUBLIC ov_asr_model)
  endif()
endif()
