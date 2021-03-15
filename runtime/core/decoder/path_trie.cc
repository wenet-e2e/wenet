// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#include "decoder/path_trie.h"

namespace wenet {

PathTrie::PathTrie() {
  prob_b_prev_ = -kFloatMax;
  prob_nb_prev_ = -kFloatMax;
  score_ = -kFloatMax;
  prob_b_cur_ = -kFloatMax;
  prob_nb_cur_ = -kFloatMax;

  exists_ = true;
  id_ = root_;
  prob_ = -kFloatMax;
  parent_ = nullptr;
  children_.clear();
  time_steps_.clear();
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child;
  }
}

bool PathTrie::PrefixCompare(const PathTrie* x, const PathTrie* y) {
  if (x->score_ == y->score_) {
    return x->id_ < y->id_;
  }
  return x->score_ > y->score_;
}

void PathTrie::Append(int id, float prob, int time_step) {
  // Case 0: * + ε => *
  if (id == blank_) {
    prob_b_cur_ = LogAdd(prob_b_cur_, score_ + prob);
    return;
  }

  auto new_prefix = GetPathTrie(id, prob, time_step);
  if (id == id_) {
    // Case 1: *a + a => *a
    prob_nb_cur_ = LogAdd(prob_nb_cur_, prob_nb_prev_ + prob);
    // Case 2: *aε + a => *aa
    new_prefix->prob_nb_cur_ =
            LogAdd(new_prefix->prob_nb_cur_, prob_b_prev_ + prob);
  } else {
    // Case 3: *a + b => *ab
    new_prefix->prob_nb_cur_ = LogAdd(new_prefix->prob_nb_cur_, score_ + prob);
  }
}

PathTrie* PathTrie::GetPathTrie(int id, float prob, int time_step) {
  for (auto& child : children_) {
    // Prefix exists, check and update
    if (child->id_ == id) {
      if (child->prob_ < prob) {
        child->prob_ = prob;
        child->time_steps_.back() = time_step;
      }
      return child;
    }
  }
  // Prefix not exists
  auto* new_path = new PathTrie();
  new_path->id_ = id;
  new_path->prob_ = prob;
  new_path->parent_ = this;
  new_path->time_steps_ = time_steps_;
  new_path->time_steps_.emplace_back(time_step);
  children_.emplace_back(new_path);
  return new_path;
}

PathTrie* PathTrie::GetPathVec(std::vector<int>* output) {
  if (id_ == root_) {
    std::reverse(output->begin(), output->end());
    return this;
  }
  output->emplace_back(id_);
  return parent_->GetPathVec(output);
}

void PathTrie::UpdatePrefixes() {
  if (exists_) {
    prob_b_prev_ = prob_b_cur_;
    prob_nb_prev_ = prob_nb_cur_;
    score_ = LogAdd(prob_b_prev_, prob_nb_prev_);
    prob_b_cur_ = -kFloatMax;
    prob_nb_cur_ = -kFloatMax;
  }
  for (auto child : children_) {
    child->UpdatePrefixes();
  }
}

void PathTrie::GetPrefixes(std::vector<PathTrie*>* output) {
  if (exists_) {
    output->emplace_back(this);
  }
  for (auto child : children_) {
    child->GetPrefixes(output);
  }
}

void PathTrie::Remove() {
  exists_ = false;
  // Current path has no children, means the path is not end with ε
  if (children_.empty()) {
    auto& children = parent_->children_;
    for (auto child = children.begin(); child != children.end(); child++) {
      if ((*child)->id_ == id_) {
        children.erase(child);
        break;
      }
    }
    // Current deleted node has no brothers
    if (children.empty() && !parent_->exists_) {
      parent_->Remove();
    }
    delete this;
  }
}

}  // namespace wenet
