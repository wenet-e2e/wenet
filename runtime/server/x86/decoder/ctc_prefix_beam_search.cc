// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"

namespace wenet {

CtcPrefixBeamSearch::CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts)
    : opts_(opts) {
  Reset();
}

void CtcPrefixBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  PrefixScore prefix_score(0.0, -std::numeric_limits<float>::max());
  std::vector<int> empty;
  cur_hyps_[empty] = prefix_score;
}

static float LogAdd(float x, float y) {
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
  float prob_a = LogAdd(a.second.s, a.second.ns);
  float prob_b = LogAdd(b.second.s, b.second.ns);
  return prob_a > prob_b;
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
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
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
          PrefixScore& next_score = next_hyps[prefix];
          next_score.s = LogAdd(next_score.s, LogAdd(prefix_score.s + prob,
                                                     prefix_score.ns + prob));
        } else if (prefix.size() > 0 && id == prefix.back()) {
          // Case 1: *aa -> *a;
          PrefixScore& next_score1 = next_hyps[prefix];
          next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
          // Case 2: *a-a -> *aa; - is blank
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = next_hyps[new_prefix];
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
        } else {
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          next_score.ns = LogAdd(next_score.ns, LogAdd(prefix_score.s + prob,
                                                       prefix_score.ns + prob));
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
    for (size_t i = 0; i < arr.size(); ++i) {
      cur_hyps_[arr[i].first] = arr[i].second;
      hypotheses_.emplace_back(std::move(arr[i].first));
      likelihood_.emplace_back(LogAdd(arr[i].second.s, arr[i].second.ns));
    }
  }
}

}  // namespace wenet
