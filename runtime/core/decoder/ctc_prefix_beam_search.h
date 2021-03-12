// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/path_trie.h"
#include "utils/utils.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

struct CtcPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
};

class CtcPrefixBeamSearch {
 public:
  explicit CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts);

  void Search(const torch::Tensor& logp);
  void Reset();

  const std::vector<std::vector<int>>& hypotheses() const {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& time_steps() const {
    return time_steps_;
  }
  const std::vector<float>& likelihood() const { return likelihood_; }

 private:
  // Nbest list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<std::vector<int>> time_steps_;
  std::vector<float> likelihood_;

  int abs_time_step;
  std::vector<PathTrie*> prefixes;
  PathTrie root;

  const CtcPrefixBeamSearchOptions& opts_;

 public:
  DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_BEAM_SEARCH_H_
