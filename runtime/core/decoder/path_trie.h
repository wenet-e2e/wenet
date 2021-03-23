// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_PATH_TRIE_H_
#define DECODER_PATH_TRIE_H_

#include <utility>
#include <vector>

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

  float score() const { return score_; }
  const std::vector<int>& time_steps() const { return time_steps_; }

  void set_prob_b_prev(const float prob_b_prev) { prob_b_prev_ = prob_b_prev; }
  void set_score(const float score) { score_ = score; }

 private:
  float prob_b_cur_;
  float prob_nb_cur_;
  // score_: LogAdd(prob_b_prev_, prob_nb_prev_)
  float prob_b_prev_;
  float prob_nb_prev_;
  float score_;

  int id_;
  float prob_;
  bool exists_;
  PathTrie* parent_;
  std::vector<PathTrie*> children_;
  std::vector<int> time_steps_;
};

}  // namespace wenet

#endif  // DECODER_PATH_TRIE_H_
