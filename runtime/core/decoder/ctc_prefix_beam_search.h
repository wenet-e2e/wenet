// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/context_graph.h"
#include "decoder/search_interface.h"
#include "utils/utils.h"

namespace wenet {

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
  int context_state = 0;
  float context_score = 0;
  std::vector<int> start_boundaries;
  std::vector<int> end_boundaries;

  void CopyContext(const PrefixScore& prefix_score) {
    context_state = prefix_score.context_state;
    context_score = prefix_score.context_score;
    start_boundaries = prefix_score.start_boundaries;
    end_boundaries = prefix_score.end_boundaries;
  }

  void UpdateContext(const std::shared_ptr<ContextGraph>& context_graph,
                     const PrefixScore& prefix_score, int word_id,
                     int prefix_len) {
    this->CopyContext(prefix_score);

    float score = 0;
    bool is_start_boundary = false;
    bool is_end_boundary = false;

    context_state =
        context_graph->GetNextState(prefix_score.context_state, word_id, &score,
                                    &is_start_boundary, &is_end_boundary);
    context_score += score;
    if (is_start_boundary) start_boundaries.emplace_back(prefix_len);
    if (is_end_boundary) end_boundaries.emplace_back(prefix_len);
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

  void Search(const std::vector<std::vector<float>>& logp) override;
  void Reset() override;
  void FinalizeSearch() override;
  SearchType Type() const override { return SearchType::kPrefixBeamSearch; }
  void UpdateOutputs(const std::pair<std::vector<int>, PrefixScore>& prefix);
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);
  void UpdateFinalContext();

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
