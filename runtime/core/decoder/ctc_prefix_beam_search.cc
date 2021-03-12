// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "lm/lm_fst.h"
#include "utils/log.h"

namespace wenet {

CtcPrefixBeamSearch::CtcPrefixBeamSearch(
    const CtcPrefixBeamSearchOptions& opts, std::shared_ptr<LmFst> lm_fst,
    std::shared_ptr<fst::SymbolTable> symbol_table)
    : opts_(opts), lm_fst_(lm_fst), symbol_table_(symbol_table) {
  Reset();
}

void CtcPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  PrefixScore prefix_score(0.0, -std::numeric_limits<float>::max());
  if (lm_fst_ != nullptr) {
    prefix_score.lm_state = lm_fst_->start();
  }
  std::vector<int> empty;
  cur_hyps_[empty] = prefix_score;
}

float LogAdd(float x, float y) {
  const float kMinLogDiffFloat = std::log(1.19209290e-7f);
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + log1p(expf(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
  return a.second.CombinedScore() > b.second.CombinedScore();
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search/
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
// TODO(Binbin Zhang): Support timestamp
void CtcPrefixBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  for (int t = 0; t < logp.size(0); ++t) {
    torch::Tensor logp_t = logp[t];
    PrefixTable next_hyps;
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); ++i) {
      int id = topk_index[i].item<int>();
      float prob = topk_score[i].item<float>();
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;
        // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
        // PrefixScore(-inf, -inf) by default, since the default constructor
        // of PrefixScore will set fields s(blank ending score) and
        // ns(none blank ending score) to -inf, respectively.
        if (id == opts_.blank) {
          PrefixScore* next_score =
              OptionalUpdateLM(prefix_score, true, prefix, &next_hyps);
          // PrefixScore& next_score = next_hyps[prefix];
          next_score->s = LogAdd(next_score->s, LogAdd(prefix_score.s + prob,
                                                       prefix_score.ns + prob));
        } else if (prefix.size() > 0 && id == prefix.back()) {
          // Case 1: *aa -> *a;
          // PrefixScore& next_score1 = next_hyps[prefix];
          PrefixScore* next_score1 =
              OptionalUpdateLM(prefix_score, true, prefix, &next_hyps);
          next_score1->ns = LogAdd(next_score1->ns, prefix_score.ns + prob);
          // Case 2: *a-a -> *aa; - is blank
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore* next_score2 =
              OptionalUpdateLM(prefix_score, false, new_prefix, &next_hyps);
          // PrefixScore& next_score2 = next_hyps[new_prefix];
          next_score2->ns = LogAdd(next_score2->ns, prefix_score.s + prob);
        } else {
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore* next_score =
              OptionalUpdateLM(prefix_score, false, new_prefix, &next_hyps);
          // PrefixScore& next_score = next_hyps[new_prefix];
          next_score->ns =
              LogAdd(next_score->ns,
                     LogAdd(prefix_score.s + prob, prefix_score.ns + prob));
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

    cur_hyps_.clear();
    hypotheses_.clear();
    likelihood_.clear();
    scores_.clear();
    for (size_t i = 0; i < arr.size(); ++i) {
      string hyp;
      for (size_t j = 0; j < arr[i].first.size(); j++) {
        hyp += symbol_table_->Find(arr[i].first[j]);
        hyp += " ";
      }
      LOG(INFO) << hyp << arr[i].second.CombinedScore() << " "
                << arr[i].second.AmScore() << " " << arr[i].second.LmScore();
      cur_hyps_[arr[i].first] = arr[i].second;
      hypotheses_.emplace_back(std::move(arr[i].first));
      likelihood_.emplace_back(arr[i].second.CombinedScore());
      scores_.push_back(arr[i].second);
    }
    LOG(INFO) << " ";
  }
}

PrefixScore* CtcPrefixBeamSearch::OptionalUpdateLM(
    const PrefixScore& prefix_score, bool copy_lm_from_prefix,
    const std::vector<int>& new_prefix, PrefixTable* next_hyps) {
  // No lm, just use the default 0 value, do nothing
  if (lm_fst_ == nullptr || opts_.lm_weight == 0.0) {
    return &(*next_hyps)[new_prefix];
  }
  if (copy_lm_from_prefix) {
    PrefixScore& next_score = (*next_hyps)[new_prefix];
    next_score.lm_state = prefix_score.lm_state;
    next_score.lm_score = prefix_score.lm_score;
    return &next_score;
  } else {
    // If this new_prefix is in cur_hyps_ or next_hyps, if in, the LM
    // info has already been updated
    float lm_score = 0.0;
    int lm_state = 0;
    if (cur_hyps_.find(new_prefix) != cur_hyps_.end()) {
      lm_state = cur_hyps_[new_prefix].lm_state;
      lm_score = cur_hyps_[new_prefix].lm_score;
    } else if (next_hyps->find(new_prefix) != next_hyps->end()) {
      lm_state = (*next_hyps)[new_prefix].lm_state;
      lm_score = (*next_hyps)[new_prefix].lm_score;
    } else {
      float lm_penalty =
          lm_fst_->Step(prefix_score.lm_state, new_prefix.back(), &lm_state);
      lm_score = prefix_score.lm_score + opts_.lm_weight * lm_penalty;
      // if (symbol_table_ != nullptr) {
      //   std::string hyp;
      //   for (size_t i = 0; i < new_prefix.size(); i++) {
      //     hyp += symbol_table_->Find(new_prefix[i]);
      //     hyp += " ";
      //   }
      //   LOG(INFO) << hyp << lm_penalty << " " << lm_score;
      // }
    }
    PrefixScore& next_score = (*next_hyps)[new_prefix];
    next_score.lm_state = lm_state;
    next_score.lm_score = lm_score;
    return &next_score;
  }
}

void CtcPrefixBeamSearch::ApplyEosScore() {
  if (lm_fst_ == nullptr || opts_.lm_weight == 0.0) return;
  for (size_t i = 0; i < scores_.size(); i++) {
    int lm_state = scores_[i].lm_state;
    float lm_penalty = lm_fst_->StepEos(scores_[i].lm_state, &lm_state);
    scores_[i].lm_score += opts_.lm_weight * lm_penalty;
    scores_[i].lm_state = lm_state;
  }
}

}  // namespace wenet
