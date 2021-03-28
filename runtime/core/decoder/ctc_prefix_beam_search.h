// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

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
  float s = -kFloatMax;               // blank endding score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank endding score
  float v_ns = -kFloatMax;            // viterbi none blank endding score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path

  PrefixScore() = default;
  float score() const { return LogAdd(s, ns); }
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
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
  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& times() const { return times_; }

 private:
  int abs_time_step_;
  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;

  // Nbest list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  const CtcPrefixBeamSearchOptions& opts_;

 public:
  DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
