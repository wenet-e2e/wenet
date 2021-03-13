// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <tuple>
#include <utility>

#include "glog/logging.h"

namespace wenet {

CtcPrefixBeamSearch::CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts)
    : opts_(opts) {
  Reset();
}

void CtcPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  time_steps_.clear();
  likelihood_.clear();

  abs_time_step_ = 0;
  // init prefixes' root
  root_.score = 0.0;
  root_.log_prob_b_prev = 0.0;
  prefixes_.emplace_back(&root_);
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in it.
void CtcPrefixBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  for (int t = 0; t < logp.size(0); t++, abs_time_step_++) {
    torch::Tensor logp_t = logp[t];
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); i++) {
      int id = topk_index[i].item<int>();
      auto prob = topk_score[i].item<float>();
      for (auto prefix : prefixes_) {
        if (id == opts_.blank) {
          prefix->log_prob_b_cur = LogAdd(prefix->log_prob_b_cur,
                                          prefix->score + prob);
          continue;
        }
        if (id == prefix->id) {
          prefix->log_prob_nb_cur = LogAdd(prefix->log_prob_nb_cur,
                                           prefix->log_prob_nb_prev + prob);
        }

        auto prefix_new = prefix->GetPathTrie(id, prob, abs_time_step_);
        if (prefix_new != nullptr) {
          float log_prob = -kFloatMax;
          if (id == prefix->id && prefix->log_prob_b_prev > -kFloatMax) {
            log_prob = prob + prefix->log_prob_b_prev;
          } else if (id != prefix->id) {
            log_prob = prob + prefix->score;
          }
          prefix_new->log_prob_nb_cur = LogAdd(prefix_new->log_prob_nb_cur,
                                               log_prob);
        }
      }
    }
    prefixes_.clear();
    root_.IterateToVec(&prefixes_);

    // 3. Second beam prune, only keep top n best paths
    if (prefixes_.size() >= opts_.second_beam_size) {
      std::nth_element(prefixes_.begin(),
                       prefixes_.begin() + opts_.second_beam_size,
                       prefixes_.end(), PathTrie::PrefixCompare);
      for (size_t i = opts_.second_beam_size; i < prefixes_.size(); i++) {
        prefixes_[i]->remove();
      }
      prefixes_.resize(opts_.second_beam_size);
    }

    // 4. Backtracking to get the beam search result each time step
    hypotheses_.clear();
    time_steps_.clear();
    likelihood_.clear();
    std::sort(prefixes_.begin(), prefixes_.end(), PathTrie::PrefixCompare);
    for (auto& prefix : prefixes_) {
      std::vector<int> hypothesis;
      std::vector<int> time_steps;
      prefix->GetPathVec(&hypothesis, &time_steps);
      hypotheses_.emplace_back(hypothesis);
      time_steps_.emplace_back(time_steps);
      likelihood_.emplace_back(prefix->score);
    }
  }
}

}  // namespace wenet
