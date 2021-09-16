// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include "ThreadPool/ThreadPool.h"
#include "ctc_beam_search_decoder.h"
#include "decoder_utils.h"
#include "fst/fstlib.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
std::vector<std::pair<double, std::vector<int>>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &log_probs_seq,
    const std::vector<std::vector<int>> &log_probs_idx, PathTrie &root,
    const bool start, size_t beam_size, int blank_id, int space_id,
    double cutoff_prob, Scorer *ext_scorer) {
  if (start) {
    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
      auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
      fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
      root.set_dictionary(dict_ptr);
      auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
      root.set_matcher(matcher);
    }
  }
  int timesteps = log_probs_seq.size();

  std::vector<PathTrie *> prefixes;

  // update log probs
  if (root.log_prob_b_prev == -NUM_FLT_INF && start) {
    root.score = root.log_prob_b_prev = 0.0;
  }
  root.iterate_to_vec_only(prefixes);
  int prev_id = -1;
  // prefix search over time
  for (size_t time_step = 0; time_step < timesteps; ++time_step) {
    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;

    auto &log_prob = log_probs_seq[time_step];
    auto &log_prob_idx = log_probs_idx[time_step];

    double top_prob = exp(log_prob[0]);
    auto top_id = log_prob_idx[0];
    if (top_prob >= cutoff_prob && top_id == blank_id) {
      if (prev_id == blank_id)
        continue;  // skip this round
      else
        prev_id = top_id;
    } else {
      prev_id = -1;
    }
    // loop over chars
    double cur_acc_prob = 0.0;
    for (size_t index = 0; index < log_prob.size(); index++) {
      auto c = log_prob_idx[index];
      float log_prob_c = log_prob[index];
      cur_acc_prob += exp(log_prob_c);
      if (cur_acc_prob > cutoff_prob && index >= 1) break;
      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c);
        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }
            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }

          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary
    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);
    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(), prefixes.begin() + beam_size,
                       prefixes.end(), prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time
  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
  return get_beam_search_result(prefixes, beam_size);
}

std::string map_sent(const std::vector<int> &sent,
                     const std::vector<std::string> &vocabulary, bool greedy,
                     int blank_id) {
  std::string output_str;

  if (!greedy) {
    for (size_t j = 0; j < sent.size(); j++) {
      output_str += vocabulary[sent[j]];
    }
  } else {
    // greedy search
    int prev = -1;
    for (size_t i = 0; i < sent.size(); i++) {
      int cur = sent[i];
      if (cur != prev && cur != blank_id) output_str += vocabulary[cur];
      prev = cur;
    }
  }
  return output_str;
}

std::vector<std::string> map_batch(
    const std::vector<std::vector<int>> &batch_sents,
    const std::vector<std::string> &vocabulary, size_t num_processes,
    bool greedy, int blank_id) {
  ThreadPool pool(num_processes);
  size_t batch_size = batch_sents.size();
  std::vector<std::future<std::string>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(map_sent, std::ref(batch_sents[i]),
                                  vocabulary, greedy, blank_id));
  }
  // get decoding results
  std::vector<std::string> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}

std::vector<std::vector<std::pair<double, std::vector<int>>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &batch_log_probs_seq,
    const std::vector<std::vector<std::vector<int>>> &batch_log_probs_idx,
    std::vector<PathTrie *> &batch_root_trie,
    const std::vector<bool> &batch_start, size_t beam_size,
    size_t num_processes, int blank_id, int space_id, double cutoff_prob,
    Scorer *ext_scorer) {
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = batch_log_probs_seq.size();

  // enqueue the tasks of decoding

  std::vector<std::future<std::vector<std::pair<double, std::vector<int>>>>>
      res;

  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(
        pool.enqueue(ctc_beam_search_decoder, std::ref(batch_log_probs_seq[i]),
                     std::ref(batch_log_probs_idx[i]),
                     std::ref(*batch_root_trie[i]), batch_start[i], beam_size,
                     blank_id, space_id, cutoff_prob, ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, std::vector<int>>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
