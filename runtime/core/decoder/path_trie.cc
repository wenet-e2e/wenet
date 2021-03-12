// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#include "decoder/path_trie.h"

#include "utils/utils.h"

namespace wenet {

PathTrie::PathTrie() {
  log_prob_b_prev = -kFloatMax;
  log_prob_nb_prev = -kFloatMax;
  log_prob_b_cur = -kFloatMax;
  log_prob_nb_cur = -kFloatMax;
  log_prob_c = -kFloatMax;
  score = -kFloatMax;
  character = -1;
  time_step = 0;
  parent = nullptr;

  ROOT_ = -1;
  exists_ = true;
  children_.clear();
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

bool PathTrie::prefix_compare(const PathTrie* x, const PathTrie* y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    }
    return x->character < y->character;
  }
  return x->score > y->score;
}

PathTrie* PathTrie::GetPathTrie(int new_char, int new_time_step,
                                float cur_log_prob_c) {
  auto child = children_.begin();
  for (; child != children_.end(); child++) {
    if (child->first == new_char) {
      if (child->second->log_prob_c < cur_log_prob_c) {
        child->second->log_prob_c = cur_log_prob_c;
        child->second->time_step = new_time_step;
      }
      if (!child->second->exists_) {
        child->second->exists_ = true;
        child->second->log_prob_b_prev = -kFloatMax;
        child->second->log_prob_nb_prev = -kFloatMax;
        child->second->log_prob_b_cur = -kFloatMax;
        child->second->log_prob_nb_cur = -kFloatMax;
      }
      return child->second;
    }
  }
  auto* new_path = new PathTrie();
  new_path->character = new_char;
  new_path->time_step = new_time_step;
  new_path->parent = this;
  new_path->log_prob_c = cur_log_prob_c;
  children_.emplace_back(new_char, new_path);
  return new_path;
}

PathTrie* PathTrie::GetPathVec(std::vector<int>* output,
                               std::vector<int>* time_steps) {
  return GetPathVec(output, time_steps, ROOT_);
}

PathTrie* PathTrie::GetPathVec(std::vector<int>* output,
                               std::vector<int>* time_steps, int stop) {
  if (character == stop || character == ROOT_) {
    std::reverse(output->begin(), output->end());
    std::reverse(time_steps->begin(), time_steps->end());
    return this;
  }
  output->emplace_back(character);
  time_steps->emplace_back(time_step);
  return parent->GetPathVec(output, time_steps, stop);
}

void PathTrie::IterateToVec(std::vector<PathTrie*>* output) {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;
    log_prob_b_cur = -kFloatMax;
    log_prob_nb_cur = -kFloatMax;
    score = LogAdd(log_prob_b_prev, log_prob_nb_prev);
    output->emplace_back(this);
  }
  for (auto child : children_) {
    child.second->IterateToVec(output);
  }
}

void PathTrie::remove() {
  exists_ = false;
  if (children_.empty()) {
    auto child = parent->children_.begin();
    for (; child != parent->children_.end(); child++) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }
    if (parent->children_.empty() && !parent->exists_) {
      parent->remove();
    }
    delete this;
  }
}

}  // namespace wenet
