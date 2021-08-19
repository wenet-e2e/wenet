// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#include "decoder/context_graph.h"

#include "fst/determinize.h"

#include "utils/string.h"

namespace wenet {

ContextGraph::ContextGraph() {}

void ContextGraph::BuildContextGraph(
    const std::vector<std::string>& query_contexts,
    const std::shared_ptr<fst::SymbolTable>& symbol_table) {
  CHECK(symbol_table != nullptr) << "Symbols table should not be nullptr!";
  symbol_table_ = symbol_table;
  if (query_contexts.empty()) {
    graph_.reset();
    return;
  }

  std::unique_ptr<fst::StdVectorFst> ofst(new fst::StdVectorFst());
  // State 0 is the start state.
  int start_state = ofst->AddState();
  // State 1 is the final state.
  int final_state = ofst->AddState();
  ofst->SetStart(start_state);
  ofst->SetFinal(final_state, fst::StdArc::Weight::One());

  LOG(INFO) << "Contexts count size: " << query_contexts.size();
  int count = 0;
  for (const auto& context : query_contexts) {
    if (context.size() > config_.max_context_length) {
      LOG(INFO) << "Skip long context: " << context;
      continue;
    }
    if (++count > config_.max_contexts) break;

    std::vector<std::string> words;
    // Split context to words by symbol table, and build the context graph.
    bool no_oov = SplitUTF8StringToWords(Trim(context), symbol_table, words);
    if (!no_oov) {
      LOG(WARNING) << "Ignore unknown word found during compilation.";
      continue;
    }
    float escape_score = 0;
    int prev_state = start_state;
    int next_state = start_state;
    for (size_t i = 0; i < words.size(); ++i) {
      int word_id = symbol_table_->Find(words[i]);
      float score = config_.context_score * UTF8StringLength(words[i]);
      next_state = (i < words.size() - 1) ? ofst->AddState() : final_state;
      // Each state has an escape arc to the start state.
      if (i > 0) {
        ofst->AddArc(prev_state, fst::StdArc(0, 0, escape_score, start_state));
      }
      ofst->AddArc(prev_state,
                   fst::StdArc(word_id, word_id, score, next_state));
      prev_state = next_state;
      escape_score -= score;
    }
  }
  std::unique_ptr<fst::StdVectorFst> det_fst(new fst::StdVectorFst());
  fst::Determinize(*ofst, det_fst.get());
  graph_ = std::move(det_fst);
}

std::pair<float, float> ContextGraph::GetNextContextStates(
    const unordered_map<StateId, float>& active_states, int word_id,
    unordered_map<StateId, float>& next_active_states) const {
  if (active_states.empty() || !graph_) return std::make_pair(0, 0);

  float partial_match_score = 0;
  float full_match_score = 0;
  for (const auto& p : active_states) {
    StateId sid = p.first;
    float score = p.second;
    for (fst::ArcIterator<fst::StdFst> aiter(*graph_, sid); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      // The ilabel equals to word id, or the arc is escape arc.
      if (arc.ilabel == word_id || arc.ilabel == 0) {
        float context_score = score + arc.weight.Value();
        partial_match_score = std::max(partial_match_score, context_score);
        if (graph_->Final(arc.nextstate) == fst::StdArc::Weight::One()) {
          full_match_score = std::max(full_match_score, context_score);
        } else {
          auto iter = next_active_states.find(arc.nextstate);
          if (iter == next_active_states.end() ||
              iter->second < context_score) {
            next_active_states[arc.nextstate] = context_score;
          }
        }
      }
    }
  }
  return std::make_pair(partial_match_score, full_match_score);
}

}  // namespace wenet
