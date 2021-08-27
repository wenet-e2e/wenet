// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
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
  float s = -kFloatMax;               // blank ending score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank ending score
  float v_ns = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path

  float score() const { return LogAdd(s, ns); }
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }

  bool has_context = false;
  ContextState context_state;
  float context_score = 0;
  std::vector<std::pair<int, int>> boundaries;

  void CopyContext(const PrefixScore& prefix_score) {
    context_state = prefix_score.context_state;
    context_score = prefix_score.context_score;
    boundaries = prefix_score.boundaries;
  }

  void UpdateContext(const std::shared_ptr<ContextGraph>& context_graph,
                     const PrefixScore& prefix_score, int word_id,
                     int prefix_len) {
    this->CopyContext(prefix_score);
    context_state =
        context_graph->GetNextState(prefix_score.context_state, word_id);
    context_score += context_state.score;
    if (context_state.is_start_boundary) {
      boundaries.emplace_back(
          std::make_pair(prefix_len - 1, context_graph->start_tag_id));
    }
    if (context_state.is_end_boundary) {
      boundaries.emplace_back(
          std::make_pair(prefix_len, context_graph->end_tag_id));
    }
  }

  float total_score() const { return score() + context_score; }
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

class CtcPrefixBeamSearch : public SearchInterface {
 public:
  explicit CtcPrefixBeamSearch(
      const CtcPrefixBeamSearchOptions& opts,
      const std::shared_ptr<ContextGraph>& context_graph = nullptr);

  void Search(const torch::Tensor& logp) override;
  void Reset() override;
  // CtcPrefixBeamSearch do nothing at FinalizeSearch
  void FinalizeSearch() override {}
  SearchType Type() const override { return SearchType::kPrefixBeamSearch; }
  void UpdateOutputs(
      const std::vector<int>& input,
      const std::vector<std::pair<int, int>>& boundaries);

  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return outputs_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  int abs_time_step_ = 0;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::shared_ptr<ContextGraph> context_graph_ = nullptr;
  // Outputs contain the hypotheses_ and tags like: <context> and </context>
  std::vector<std::vector<int>> outputs_;
  const CtcPrefixBeamSearchOptions& opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
