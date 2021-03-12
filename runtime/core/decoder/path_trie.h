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
  static bool prefix_compare(const PathTrie* x, const PathTrie* y);

  // Get new prefix after appending new char
  PathTrie* GetPathTrie(int new_char, int new_time_step, float log_prob_c);
  // Get the prefix in index from root to current node
  PathTrie* GetPathVec(std::vector<int>* output, std::vector<int>* time_steps);
  // Get the prefix in index from some stop node to current node
  PathTrie* GetPathVec(std::vector<int>* output, std::vector<int>* time_steps,
                       int stop);
  // Update log probs
  void IterateToVec(std::vector<PathTrie*>* output);

  bool is_empty() const { return ROOT_ == character; }
  // Remove current path from root
  void remove();

  float log_prob_b_prev;
  float log_prob_nb_prev;
  float log_prob_b_cur;
  float log_prob_nb_cur;
  float log_prob_c;
  float score;
  int character;
  int time_step;
  PathTrie* parent;

 private:
  int ROOT_;
  bool exists_;
  std::vector<std::pair<int, PathTrie*>> children_;
};

}  // namespace wenet

#endif  // DECODER_PATH_TRIE_H_
