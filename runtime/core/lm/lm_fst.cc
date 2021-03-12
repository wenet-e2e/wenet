// Copyright (c) 2021 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lm/lm_fst.h"

#include <string>
#include <vector>

#include "utils/log.h"

namespace wenet {

LmFst::LmFst(const std::string& fst_file,
             std::shared_ptr<fst::SymbolTable> symbol_table)
    : symbol_table_(symbol_table) {
  fst_.reset(fst::StdVectorFst::Read(fst_file));
  CHECK_NE(fst_->Properties(fst::kILabelSorted, true), 0);
  sos_ = symbol_table_->Find("<s>");
  eos_ = symbol_table_->Find("</s>");
  // fst::kNoSymbol = -1
  CHECK_NE(sos_, fst::SymbolTable::kNoSymbol);
  CHECK_NE(eos_, fst::SymbolTable::kNoSymbol);
  Step(fst_->Start(), sos_, &start_);
  LOG(INFO) << "Start id step by <s> is " << start_;
}

float LmFst::Step(int state, int ilabel, int* next_state) {
  CHECK_NE(ilabel, 0);
  fst::SortedMatcher<fst::StdVectorFst> sm(*fst_, fst::MATCH_INPUT);
  sm.SetState(state);
  if (sm.Find(ilabel)) {
    const fst::StdArc& arc = sm.Value();
    *next_state = arc.nextstate;
    return -arc.weight.Value();
  }

  // Backoff
  fst::ArcIterator<fst::StdVectorFst> aiter(*fst_, state);
  // The state must has arcs and has backoff arc
  if (!aiter.Done() && aiter.Value().ilabel == 0) {
    const fst::StdArc& arc = aiter.Value();
    return -arc.weight.Value() + Step(arc.nextstate, ilabel, next_state);
  } else {
    *next_state = start_;
    // Give a weight small enough, we don't directly use float max here to
    // avoid arithmetic overflow when combined with other score outside
    return -1e5;
  }
}

float LmFst::StepEos(int state, int* next_state) {
  return Step(state, eos_, next_state);
}

float LmFst::StepTokenArray(const std::vector<std::string>& strs) {
  int state = start_;
  int next_state = 0;
  float sentence_weight = 0.0;
  std::vector<std::string> strs_add_eos(strs);
  strs_add_eos.emplace_back("</s>");
  for (size_t i = 0; i < strs_add_eos.size(); i++) {
    int ilabel = symbol_table_->Find(strs_add_eos[i]);
    CHECK_NE(ilabel, fst::SymbolTable::kNoSymbol);
    float weight = Step(state, ilabel, &next_state);
    sentence_weight += weight;
    LOG(INFO) << state << " " << next_state << " " << ilabel << "("
              << strs_add_eos[i] << ") " << weight << " " << sentence_weight;
    state = next_state;
  }
  LOG(INFO) << "Sentence weight " << sentence_weight;
  return sentence_weight;
}

}  // namespace wenet
