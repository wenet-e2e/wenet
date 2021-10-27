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

class ContextGraph {
 public:
  explicit ContextGraph(ContextConfig config);
  void BuildContextGraph(const std::vector<std::string>& query_context,
                         const std::shared_ptr<fst::SymbolTable>& symbol_table);
  int GetNextState(int cur_state, int word_id, float* score,
                   bool* is_start_boundary, bool* is_end_boundary);

  int start_tag_id() { return start_tag_id_; }
  int end_tag_id() { return end_tag_id_; }

 private:
  int start_tag_id_ = -1;
  int end_tag_id_ = -1;
  ContextConfig config_;
  std::shared_ptr<fst::SymbolTable> symbol_table_ = nullptr;
  std::unique_ptr<fst::StdVectorFst> graph_ = nullptr;
  DISALLOW_COPY_AND_ASSIGN(ContextGraph);
};

}  // namespace wenet

#endif  // DECODER_CONTEXT_GRAPH_H_
