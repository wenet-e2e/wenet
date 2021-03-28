// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_TORCH_ASR_DECODER_H_
#define DECODER_TORCH_ASR_DECODER_H_

#include <memory>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/symbol_table.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;

struct DecodeOptions {
  int chunk_size = 16;
  int num_left_chunks = -1;
  CtcPrefixBeamSearchOptions ctc_search_opts;
};

// Torch ASR decoder
class TorchAsrDecoder {
 public:
  TorchAsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
                  std::shared_ptr<TorchAsrModel> model,
                  const SymbolTable& symbol_table, const DecodeOptions& opts);

  // Return true if all feature has been decoded, else return false
  bool Decode();
  void Reset();
  int num_frames_in_current_chunk() const {
    return num_frames_in_current_chunk_;
  }
  std::string result() const { return result_; }

 private:
  // Return true if we reach the end of the feature pipeline
  bool AdvanceDecoding();
  void AttentionRescoring();

  std::shared_ptr<FeaturePipeline> feature_pipeline_;
  std::shared_ptr<TorchAsrModel> model_;
  const SymbolTable& symbol_table_;
  const DecodeOptions& opts_;
  // cache feature
  std::vector<std::vector<float>> cached_feature_;
  bool start_ = false;

  torch::jit::IValue subsampling_cache_;
  // transformer/conformer encoder layers output cache
  torch::jit::IValue elayers_output_cache_;
  torch::jit::IValue conformer_cnn_cache_;
  std::vector<torch::Tensor> encoder_outs_;
  int offset_ = 0;  // offset

  std::unique_ptr<CtcPrefixBeamSearch> ctc_prefix_beam_searcher_;

  int num_frames_in_current_chunk_;
  std::string result_;

 public:
  DISALLOW_COPY_AND_ASSIGN(TorchAsrDecoder);
};

}  // namespace wenet

#endif  // DECODER_TORCH_ASR_DECODER_H_
