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

  abs_time_step = 0;
  // init prefixes' root
  root.score = 0.0;
  root.log_prob_b_prev = 0.0;
  prefixes.emplace_back(&root);
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in it.
void CtcPrefixBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  for (int t = 0; t < logp.size(0); t++, abs_time_step++) {
    torch::Tensor logp_t = logp[t];
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); i++) {
      int id = topk_index[i].item<int>();
      auto prob = topk_score[i].item<float>();
      for (auto prefix : prefixes) {
        if (id == opts_.blank) {
          prefix->log_prob_b_cur = LogAdd(prefix->log_prob_b_cur,
                                          prefix->score + prob);
          continue;
        }
        if (id == prefix->character) {
          prefix->log_prob_nb_cur = LogAdd(prefix->log_prob_nb_cur,
                                           prefix->log_prob_nb_prev + prob);
        }

        auto prefix_new = prefix->GetPathTrie(id, abs_time_step, prob);
        if (prefix_new != nullptr) {
          float log_prob = -kFloatMax;
          if (id == prefix->character && prefix->log_prob_b_prev > -kFloatMax) {
            log_prob = prob + prefix->log_prob_b_prev;
          } else if (id != prefix->character) {
            log_prob = prob + prefix->score;
          }
          prefix_new->log_prob_nb_cur = LogAdd(prefix_new->log_prob_nb_cur,
                                               log_prob);
        }
      }
    }
    prefixes.clear();
    root.IterateToVec(&prefixes);

    // 3. Second beam prune, only keep top n best paths
    if (prefixes.size() >= opts_.second_beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + opts_.second_beam_size,
                       prefixes.end(), PathTrie::prefix_compare);
      for (size_t i = opts_.second_beam_size; i < prefixes.size(); i++) {
        prefixes[i]->remove();
      }
      prefixes.resize(opts_.second_beam_size);
    }

    // 4. Backtracking to get the beam search result each time step
    hypotheses_.clear();
    time_steps_.clear();
    likelihood_.clear();
    std::sort(prefixes.begin(), prefixes.end(), PathTrie::prefix_compare);
    for (auto& prefix : prefixes) {
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
