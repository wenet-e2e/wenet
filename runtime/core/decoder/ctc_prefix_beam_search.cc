// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"

namespace wenet {

CtcPrefixBeamSearch::CtcPrefixBeamSearch(
    const CtcPrefixBeamSearchOptions& opts,
    const std::shared_ptr<ContextGraph>& context_graph)
    : opts_(opts), context_graph_(context_graph) {
  Reset();
}

void CtcPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  outputs_.clear();
  abs_time_step_ = 0;
  PrefixScore prefix_score;
  prefix_score.s = 0.0;
  prefix_score.ns = -kFloatMax;
  prefix_score.v_s = 0.0;
  prefix_score.v_ns = 0.0;
  std::vector<int> empty;
  cur_hyps_[empty] = prefix_score;
}

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
  return a.second.total_score() > b.second.total_score();
}

void CtcPrefixBeamSearch::UpdateOutputs(
    const std::pair<std::vector<int>, PrefixScore>& prefix) {
  const std::vector<int>& input = prefix.first;
  const std::vector<int>& start_boundaries = prefix.second.start_boundaries;
  const std::vector<int>& end_boundaries = prefix.second.end_boundaries;

  std::vector<int> output;
  int s = 0;
  int e = 0;
  for (int i = 0; i < input.size(); ++i) {
    if (s < start_boundaries.size() && i == start_boundaries[s]) {
      output.emplace_back(context_graph_->start_tag_id());
      ++s;
    }
    output.emplace_back(input[i]);
    if (e < end_boundaries.size() && i == end_boundaries[e]) {
      output.emplace_back(context_graph_->end_tag_id());
      ++e;
    }
  }
  outputs_.emplace_back(output);
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
void CtcPrefixBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  for (int t = 0; t < logp.size(0); ++t, ++abs_time_step_) {
    torch::Tensor logp_t = logp[t];
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); ++i) {
      int id = topk_index[i].item<int>();
      auto prob = topk_score[i].item<float>();
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;
        // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
        // PrefixScore(-inf, -inf) by default, since the default constructor
        // of PrefixScore will set fields s(blank ending score) and
        // ns(none blank ending score) to -inf, respectively.
        if (id == opts_.blank) {
          // Case 0: *a + ε => *a
          PrefixScore& next_score = next_hyps[prefix];
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          next_score.v_s = prefix_score.viterbi_score() + prob;
          next_score.times_s = prefix_score.times();
          // Prefix not changed, copy the context from prefix.
          if (context_graph_ && !next_score.has_context) {
            next_score.CopyContext(prefix_score);
            next_score.has_context = true;
          }
        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          PrefixScore& next_score1 = next_hyps[prefix];
          next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
          if (next_score1.v_ns < prefix_score.v_ns + prob) {
            next_score1.v_ns = prefix_score.v_ns + prob;
            if (next_score1.cur_token_prob < prob) {
              next_score1.cur_token_prob = prob;
              next_score1.times_ns = prefix_score.times_ns;
              CHECK_GT(next_score1.times_ns.size(), 0);
              next_score1.times_ns.back() = abs_time_step_;
            }
          }
          if (context_graph_ && !next_score1.has_context) {
            next_score1.CopyContext(prefix_score);
            next_score1.has_context = true;
          }

          // Case 2: *aε + a => *aa
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = next_hyps[new_prefix];
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
          if (next_score2.v_ns < prefix_score.v_s + prob) {
            next_score2.v_ns = prefix_score.v_s + prob;
            next_score2.cur_token_prob = prob;
            next_score2.times_ns = prefix_score.times_s;
            next_score2.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score2.has_context) {
            // Prefix changed, calculate the context score.
            next_score2.UpdateContext(context_graph_, prefix_score, id,
                                      prefix.size());
            next_score2.has_context = true;
          }
        } else {
          // Case 3: *a + b => *ab, *aε + b => *ab
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);
          if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
            next_score.v_ns = prefix_score.viterbi_score() + prob;
            next_score.cur_token_prob = prob;
            next_score.times_ns = prefix_score.times();
            next_score.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score.has_context) {
            // Calculate the context score.
            next_score.UpdateContext(context_graph_, prefix_score, id,
                                     prefix.size());
            next_score.has_context = true;
          }
        }
      }
    }

    // 3. Second beam prune, only keep top n best paths
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                              next_hyps.end());
    int second_beam_size =
        std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
    std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                     PrefixScoreCompare);
    arr.resize(second_beam_size);
    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

    // 4. Update cur_hyps_ and get new result
    cur_hyps_.clear();
    outputs_.clear();
    hypotheses_.clear();
    likelihood_.clear();
    viterbi_likelihood_.clear();
    times_.clear();
    for (auto& item : arr) {
      cur_hyps_[item.first] = item.second;
      UpdateOutputs(item);
      hypotheses_.emplace_back(std::move(item.first));
      likelihood_.emplace_back(item.second.total_score());
      viterbi_likelihood_.emplace_back(item.second.viterbi_score());
      times_.emplace_back(item.second.times());
    }
  }
}

}  // namespace wenet
