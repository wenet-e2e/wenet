// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_PATH_TRIE_H_
#define DECODER_PATH_TRIE_H_

#include <utility>
#include <vector>

#include "glog/logging.h"

#include "utils/utils.h"

namespace wenet {

/* Trie tree for prefix storing and manipulating
 */
class PathTrie {
 public:
  static const int blank_ = 0;
  static const int root_ = -1;

  PathTrie();
  ~PathTrie();

  // Functor for prefix comparison
  static bool PrefixCompare(const PathTrie* x, const PathTrie* y);
  // Append new id to prefix
  void Append(int id, float prob, int time_step);
  // Get new prefix after appending new id
  PathTrie* GetPathTrie(int id, float prob, int time_step);
  // Get the prefix in index from root to current node
  PathTrie* GetPathVec(std::vector<int>* output);
  // Update probs and score of prefixes
  void UpdatePrefixes();
  // Get prefixes
  void GetPrefixes(std::vector<PathTrie*>* output);
  // Remove current path from root for beam prune
  void Remove();

  // Must be called after UpdatePrefixes()
  float score() const { return score_; }
  float viterbi_score() { return viterbi_score_; }
  const std::vector<int>& time_steps() const { return time_steps_; }

  // Only used for root node initilization
  void InitRoot() {
    prob_b_prev_ = 0.0;
    score_ = 0.0;
    viterbi_prob_b_prev_ = 0.0;
    viterbi_prob_nb_prev_ = 0.0;
    viterbi_score_ = 0.0;
  }

 private:
  float prob_b_cur_;
  float prob_nb_cur_;
  // score_: LogAdd(prob_b_prev_, prob_nb_prev_)
  float prob_b_prev_;
  float prob_nb_prev_;
  float score_;

  // use viterbi to trace the best alignment path
  float viterbi_prob_b_cur_;   // viterbi prob end with blank
  float viterbi_prob_nb_cur_;  // viterbi prob end with none blank
  float viterbi_score_;
  float viterbi_prob_b_prev_;
  float viterbi_prob_nb_prev_;
  float cur_token_prob_;
  std::vector<int> time_steps_b_prev_;
  std::vector<int> time_steps_nb_prev_;
  std::vector<int> time_steps_b_cur_;
  std::vector<int> time_steps_nb_cur_;
  std::vector<int> time_steps_;

  int id_;
  bool exists_;
  PathTrie* parent_;
  std::vector<PathTrie*> children_;
};

}  // namespace wenet

#endif  // DECODER_PATH_TRIE_H_
