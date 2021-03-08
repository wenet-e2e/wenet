// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <limits>
#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "lm/lm_fst.h"
#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

struct CtcPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
  float lm_weight = 0.0;
};

float LogAdd(float x, float y);

struct PrefixScore {
  // blank endding score
  float s = -std::numeric_limits<float>::max();
  // none blank ending score
  float ns = -std::numeric_limits<float>::max();

  // LM info
  int lm_state = 0;
  float lm_score = 0.0f;
  PrefixScore() {}
  PrefixScore(float s, float ns) : s(s), ns(ns) {}
  float Score() const { return LogAdd(s, ns) + lm_score; }
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
  typedef std::unordered_map<std::vector<int>, PrefixScore, PrefixHash>
      PrefixTable;

  explicit CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts,
                               std::shared_ptr<LmFst> lm_fst = nullptr);

  void Search(const torch::Tensor& logp);
  void Reset();

  const std::vector<std::vector<int>>& hypotheses() const {
    return hypotheses_;
  }
  const std::vector<float>& likelihood() const { return likelihood_; }

 private:
  // Optional update LM score
  // return the reference in next_hyps
  PrefixScore& OptionalUpdateLM(const PrefixScore& prefix,
                                bool copy_lm_from_prefix,
                                const std::vector<int>& new_prefix,
                                PrefixTable* next_hyps);

  PrefixTable cur_hyps_;

  // Nbest list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::shared_ptr<LmFst> lm_fst_;
  const CtcPrefixBeamSearchOptions& opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
