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

#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>
#include "path_trie.h"
#include "scorer.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     log_probs_seq: 2-D vector that each element is a vector of log
 probabilities
 *                    for one time step, it is sorted   (topk)
 *     log_probs_idx: 2-D vector that the index of every element in
 log_probs_seq
 *                     topk index
 *     root: A PathTrie root
 *     start: whether this the first chunk of this sequence
 *     beam_size: The width of beam search.
 *     blank_id: default is 0
 *     space_id: default is -1
 *     cutoff_prob: Cutoff probability for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/
std::vector<std::pair<double, std::vector<int>>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &log_probs_seq,
    const std::vector<std::vector<int>> &log_probs_idx, PathTrie &root,
    const bool start, size_t beam_size, int blank_id = 0, int space_id = -1,
    double cutoff_prob = 0.999, Scorer *ext_scorer = nullptr);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     batch_log_probs_seq: 3-D vector that each element is a 2-D vector that
 can be used
 *                by ctc_beam_search_decoder().
 *     batch_log_probs_idx: 3-D vector that each element is a 2-D vector that
 can be used
 *                by ctc_beam_search_decoder().
 *     batch_root_trie: a batch of Path trie for each sequence
 *     batch_start: a batch of boolean value to indicate whether this is the
 first
 *                  chunk of each sequence
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     blank_id: default blank_id is 0
 *     space_id: default space_id is -1, this is for word based scorer
 *     cutoff_prob: Cutoff probability for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, std::vector<int>>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &batch_log_probs_seq,
    const std::vector<std::vector<std::vector<int>>> &batch_log_probs_idx,
    std::vector<PathTrie *> &batch_root_trie,
    const std::vector<bool> &batch_start, size_t beam_size,
    size_t num_processes, int blank_id = 0, int space_id = -1,
    double cutoff_prob = 0.999, Scorer *ext_scorer = nullptr);

/* Map vector of int to string

 * Parameters:
 *   sent: a vector of int ids
 *   vocabulary: vocabulary
 * Return:
 *   A decoded string
*/
std::string map_sent(const std::vector<int> &sent,
                     const std::vector<std::string> &vocabulary,
                     bool greedy = false, int blank_id = 0);

/* Map batch vector of int to string

 * Parameters:
 *   batch_sents: a batch of vector of int ids
 *   vocabulary: vocabulary
 *   num_processes: number of processes to use
 * Return:
 *   A vector decoded string
*/
std::vector<std::string> map_batch(
    const std::vector<std::vector<int>> &batch_sents,
    const std::vector<std::string> &vocabulary, size_t num_processes,
    bool greedy = false, int blank_id = 0);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
