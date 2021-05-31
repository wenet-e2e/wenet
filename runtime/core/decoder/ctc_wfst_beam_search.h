// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_WFST_BEAM_SEARCH_H_
#define DECODER_CTC_WFST_BEAM_SEARCH_H_

#include <memory>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/search_interface.h"
#include "kaldi/decoder/lattice-faster-online-decoder.h"
#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

class DecodableTensorScaled : public kaldi::DecodableInterface {
 public:
  explicit DecodableTensorScaled(float scale = 1.0) : scale_(scale) { Reset(); }

  void Reset();
  int32 NumFramesReady() const override { return num_frames_ready_; }
  bool IsLastFrame(int32 frame) const override;
  float LogLikelihood(int32 frame, int32 index) override;
  int32 NumIndices() const override;
  void AcceptLoglikes(const torch::Tensor& logp);
  void SetFinish() { done_ = true; }

 private:
  int num_frames_ready_ = 0;
  float scale_ = 1.0;
  bool done_ = false;
  torch::Tensor logp_;
  std::unique_ptr<torch::TensorAccessor<float, 1>> accessor_;
};

// LatticeFasterDecoderConfig has the following key members
// beam: decoding beam
// max_active: Decoder max active states
// lattice_beam: Lattice generation beam
struct CtcWfstBeamSearchOptions : public kaldi::LatticeFasterDecoderConfig {
  float acoustic_scale = 1.0;
  float nbest = 10;
  // When blank score is greater than this thresh, skip the frame in viterbi
  // search
  float blank_skip_thresh = 0.98;
};

class CtcWfstBeamSearch : public SearchInterface {
 public:
  explicit CtcWfstBeamSearch(const fst::Fst<fst::StdArc>& fst,
                             const CtcWfstBeamSearchOptions& opts);
  void Search(const torch::Tensor& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::kWfstBeamSearch; }
  // For CTC prefix beam search, both inputs and outputs are hypotheses_
  const std::vector<std::vector<int>>& Inputs() const override {
    return inputs_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  // Sub one and remove <blank>
  void ConvertToInputs(const std::vector<int>& alignment,
                       std::vector<int>* input);

  int num_frames_ = 0;
  std::vector<int> decoded_frames_mapping_;

  torch::Tensor last_frame_prob_;
  bool is_last_frame_blank_ = false;
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<float> likelihood_;
  std::vector<std::vector<int>> times_;
  DecodableTensorScaled decodable_;
  kaldi::LatticeFasterOnlineDecoder decoder_;
  const CtcWfstBeamSearchOptions& opts_;
};

}  // namespace wenet

#endif  // DECODER_CTC_WFST_BEAM_SEARCH_H_
