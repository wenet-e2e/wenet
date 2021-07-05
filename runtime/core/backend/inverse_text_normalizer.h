// Copyright (c) 2021.
// Author: sxc19@mails.tsinghua.edu.cn (Xingchen Song)
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

#ifndef BACKEND_INVERSE_TEXT_NORMALIZER_H_
#define BACKEND_INVERSE_TEXT_NORMALIZER_H_

#include <fst/compat.h>
#include <fst/vector-fst.h>
#include <fst/arc.h>
#include <fst/fst.h>
#include <fst/string.h>
#include <fst/symbol-table.h>
#include <thrax/compat/compat.h>
#include <thrax/grm-manager.h>
#include <thrax/algo/paths.h>
#include <thrax/compat/utils.h>
#include <thrax/symbols.h>

#include <string>
#include <vector>
#include <utility>
#include <memory>

#include "utils/utils.h"

enum ItnTokenType { SYMBOL = 1, BYTE = 2, UTF8 = 3 };

namespace wenet {

class InverseTextNormalizer {
  typedef fst::StringCompiler<fst::StdArc> Compiler;

 public:
  InverseTextNormalizer() = default;

  void Initialize(const std::string& far_path,
                  const std::string& rules,
                  const std::string& input_mode,
                  const std::string& output_mode);
  // Runs the input through the FSTs.
  const string ProcessInput(const std::string& input);
  // Init generated_symlab_ from fsts in grm_.
  void GetGeneratedSymbolTable();
  // Make sure ilabel and olabel are non-negative values.
  void FormatFst(fst::StdVectorFst* vfst);
  // Computes the n-shortest paths and returns a vector of strings, each string
  // corresponding to each path. The mapping of labels to strings is controlled
  // by the type and the symtab. Elements that are in the generated label set
  // from the grammar are output as "[name]" where "name" is the name of the
  // generated label. Paths are sorted in ascending order of weights.
  bool FstToStrings(const fst::VectorFst<fst::StdArc>& vfst, const int& n,
                    std::vector<std::pair<string, float> >* strings);
  // Add label to path
  bool AppendLabel(const fst::StdArc::Label& label, std::string* path);

 private:
  thrax::GrmManagerSpec<fst::StdArc> grm_;
  ItnTokenType type_ = BYTE;
  std::vector<string> rules_;
  std::shared_ptr<Compiler> compiler_ = nullptr;
  std::shared_ptr<fst::SymbolTable> generated_symtab_ = nullptr;
  std::shared_ptr<fst::SymbolTable> output_symtab_ = nullptr;
  std::shared_ptr<fst::SymbolTable> byte_symtab_ = nullptr;
  std::shared_ptr<fst::SymbolTable> utf8_symtab_ = nullptr;
  std::shared_ptr<fst::SymbolTable> input_symtab_ = nullptr;

  WENET_DISALLOW_COPY_AND_ASSIGN(InverseTextNormalizer);
};  // class InverseTextNormalizer

}  // namespace wenet

#endif  // BACKEND_INVERSE_TEXT_NORMALIZER_H_
