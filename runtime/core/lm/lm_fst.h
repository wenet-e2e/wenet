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

#ifndef LM_LM_FST_H_
#define LM_LM_FST_H_

#include <memory>

#include "fst/fstlib.h"
#include "fst/symbol-table.h"

namespace wenet {

const std::string kSOS = "<s>";
const std::string kEOS = "</s>";

class LmFst {
 public:
  LmFst(const std::string& fst_file, const std::string& symbol_file);
  // Process given state with given label, return weight and the next state
  float Step(int state, int ilabel, int* next_state);
  float StepEos(int state, int* next_state);
  float StepTokenArray(std::vector<std::string>& strs);

  int start() const { return start_; }

 private:
  std::unique_ptr<fst::StdVectorFst> fst_ = nullptr;
  std::unique_ptr<fst::SymbolTable> symbols_ = nullptr;

  int sos_ = -1;
  int eos_ = -1;
  int start_ = -1;
};

}  // namespace wenet

#endif  // LM_LM_FST_H_
