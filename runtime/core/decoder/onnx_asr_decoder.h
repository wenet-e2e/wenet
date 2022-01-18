// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
// Copyright 2021 Huya Inc. All Rights Reserved.
// Author: lizexuan@huya.com (Zexuan Li)

#ifndef DECODER_ONNX_ASR_DECODER_H_
#define DECODER_ONNX_ASR_DECODER_H_

#include "decoder/torch_asr_decoder.h"

namespace wenet {

// Onnx ASR decoder
class OnnxAsrDecoder {
 public:
  OnnxAsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
                 std::shared_ptr<DecodeResource> resource,
                 const DecodeOptions& opts);

  DecodeState Decode();
  void Rescoring();
  void Reset();
  void ResetContinuousDecoding();
  bool DecodedSomething() const {
    return !result_.empty() && !result_[0].sentence.empty();
  }

  // This method is used for time benchmark
  int num_frames_in_current_chunk() const {
    return num_frames_in_current_chunk_;
  }
  int frame_shift_in_ms() const {
    return model_->subsampling_rate() *
           feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  int feature_frame_shift_in_ms() const {
    return feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  const std::vector<DecodeResult>& result() const { return result_; }

 private:
  // Return true if we reach the end of the feature pipeline
  DecodeState AdvanceDecoding();
  void AttentionRescoring();
  void OnnxReset();
  float AttentionDecoderScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);
  void UpdateResult(bool finish = false);

  std::shared_ptr<FeaturePipeline> feature_pipeline_;
  std::shared_ptr<OnnxAsrModel> model_;
  std::shared_ptr<PostProcessor> post_processor_;

  std::shared_ptr<fst::Fst<fst::StdArc>> fst_ = nullptr;
  // output symbol table
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  // e2e unit symbol table
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
  const DecodeOptions& opts_;
  // cache feature
  std::vector<std::vector<float>> cached_feature_;
  bool start_ = false;

  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const char* input_names_[6] = {"input", "offset", "required_cache_size",
                                 "i1",    "i2",     "i3"};
  const char* output_names_[4] = {"output", "o1", "o2", "o3"};
  const char* decode_input_names_[3] = {"hyps_pad", "hyps_lens", "encoder_out"};
  const char* decode_output_names_[2] = {"o1", "o2"};

  const char* ctc_input_names_[1] = {"input"};
  const char* ctc_output_names_[1] = {"output"};

  Ort::Value subsampling_cache_ort_{nullptr};
  Ort::Value elayers_output_cache_ort_{nullptr};
  Ort::Value conformer_cnn_cache_ort_{nullptr};
  std::vector<Ort::Value> encoder_outs_ort_;
  std::vector<float> subsampling_cache_;
  std::vector<float> elayers_output_cache_;
  std::vector<float> conformer_cnn_cache_;

  int offset_ = 1;  // offset
  // For continuous decoding
  int num_frames_ = 0;
  int global_frame_offset_ = 0;

  std::unique_ptr<SearchInterface> searcher_;
  std::unique_ptr<CtcEndpoint> ctc_endpointer_;

  int num_frames_in_current_chunk_ = 0;
  std::vector<DecodeResult> result_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(OnnxAsrDecoder);
};

}  // namespace wenet

#endif  // DECODER_TORCH_ASR_DECODER_H_
