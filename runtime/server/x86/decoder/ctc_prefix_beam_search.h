// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <limits>
#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

struct CtcPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
};

struct PrefixScore {
  // blank endding score
  float s = -std::numeric_limits<float>::max();
  // none blank ending score
  float ns = -std::numeric_limits<float>::max();
  PrefixScore() {}
  PrefixScore(float s, float ns) : s(s), ns(ns) {}
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (size_t i = 0; i < prefix.size(); ++i) {
      hash_code = prefix[i] + 31 * hash_code;
    }
    return hash_code;
  }
};

class CtcPrefixBeamSearch {
 public:
  explicit CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts);

  void Search(const torch::Tensor& logp);
  void Reset();

  const std::vector<std::vector<int>>& hypotheses() const {
    return hypotheses_;
  }
  const std::vector<float>& likelihood() const { return likelihood_; }

 private:
  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;

  // Nbest list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;

  const CtcPrefixBeamSearchOptions& opts_;

 public:
  DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
