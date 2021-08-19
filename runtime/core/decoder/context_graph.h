// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_CONTEXT_GRAPH_H_
#define DECODER_CONTEXT_GRAPH_H_

#include "fst/compose.h"
#include "fst/fst.h"
#include "fst/vector-fst.h"

namespace wenet {

struct ContextConfig {
  int max_contexts = 5000;
  int max_context_length = 100;
  float context_score = 3.0;
};

class ContextGraph {
  using StateId = fst::StdArc::StateId;

 public:
  ContextGraph();
  void BuildContextGraph(const std::vector<std::string>& query_context,
                         const std::shared_ptr<fst::SymbolTable>& symbol_table);
  // Return the partial match score and the full match score.
  std::pair<float, float> GetNextContextStates(
      const unordered_map<StateId, float>& active_states, int word_id,
      unordered_map<StateId, float>& next_active_states) const;

  ContextConfig config_;

 private:
  std::shared_ptr<fst::SymbolTable> symbol_table_ = nullptr;
  std::unique_ptr<fst::StdVectorFst> graph_ = nullptr;
  DISALLOW_COPY_AND_ASSIGN(ContextGraph);
};

}  // namespace wenet

#endif  // DECODER_CONTEXT_GRAPH_H_
