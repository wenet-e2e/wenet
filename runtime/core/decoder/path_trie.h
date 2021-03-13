// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_PATH_TRIE_H_
#define DECODER_PATH_TRIE_H_

#include <vector>
#include <utility>

namespace wenet {

/* Trie tree for prefix storing and manipulating
 */
class PathTrie {
 public:
  PathTrie();
  ~PathTrie();

  // Functor for prefix comparison
  static bool PrefixCompare(const PathTrie* x, const PathTrie* y);
  // Get new prefix after appending new id
  PathTrie* GetPathTrie(int new_id, float new_log_prob, int new_time_step);
  // Get the prefix in index from root to current node
  PathTrie* GetPathVec(std::vector<int>* output, std::vector<int>* time_steps);
  // Update log probs
  void IterateToVec(std::vector<PathTrie*>* output);
  bool empty() const { return id == root_; }
  // Remove current path from root
  void remove();

  int id;
  float log_prob;
  int time_step;

  float log_prob_b_prev;
  float log_prob_nb_prev;
  float log_prob_b_cur;
  float log_prob_nb_cur;
  float score;
  PathTrie* parent;

 private:
  const int root_ = -1;
  bool exists_;
  std::vector<std::pair<int, PathTrie*>> children_;
};

}  // namespace wenet

#endif  // DECODER_PATH_TRIE_H_
