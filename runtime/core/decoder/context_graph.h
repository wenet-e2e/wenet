// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_CONTEXT_GRAPH_H_
#define DECODER_CONTEXT_GRAPH_H_

#include <memory>
#include <string>
#include <vector>

#include "fst/compose.h"
#include "fst/fst.h"
#include "fst/vector-fst.h"

namespace wenet {

using StateId = fst::StdArc::StateId;

struct ContextConfig {
  int max_contexts = 5000;
  int max_context_length = 100;
  float context_score = 3.0;
};

struct ContextState {
  StateId sid = 0;
  float score = 0.0;
  // A context state could be both start boundary and end boundary.
  bool is_start_boundary = false;
  bool is_end_boundary = false;
};

class ContextGraph {
 public:
  ContextGraph();
  void BuildContextGraph(const std::vector<std::string>& query_context,
                         const std::shared_ptr<fst::SymbolTable>& symbol_table);
  ContextState GetNextState(const ContextState& cur_state, int word_id);

  ContextConfig config_;
  int start_tag_id = -1;
  int end_tag_id = -1;

 private:
  std::shared_ptr<fst::SymbolTable> symbol_table_ = nullptr;
  std::unique_ptr<fst::StdVectorFst> graph_ = nullptr;
  DISALLOW_COPY_AND_ASSIGN(ContextGraph);
};

}  // namespace wenet

#endif  // DECODER_CONTEXT_GRAPH_H_
