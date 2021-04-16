// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_WFST_BEAM_SEARCH_H_
#define DECODER_CTC_WFST_BEAM_SEARCH_H_

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
  void AcceptPosterior(const torch::Tensor& logp);
  void SetFinish() { done_ = true; }

 private:
  int offset_ = 0;
  int num_frames_ready_ = 0;
  float scale_ = 1.0;
  bool done_ = false;
  torch::Tensor logp_;
};

class CtcWfstBeamSearch : public SearchInterface {
 public:
  explicit CtcWfstBeamSearch(const fst::Fst<fst::StdArc>& fst,
                             const kaldi::LatticeFasterDecoderConfig& opts);
  void Search(const torch::Tensor& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
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
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<float> likelihood_;
  std::vector<std::vector<int>> times_;
  DecodableTensorScaled decodable_;
  kaldi::LatticeFasterOnlineDecoder decoder_;
};

}  // namespace wenet

#endif  // DECODER_CTC_WFST_BEAM_SEARCH_H_
