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

#include "inverse_text_normalizer.h"

#include <set>

#include "utils/log.h"
#include "utils/flags.h"

namespace wenet {

void InverseTextNormalizer::Initialize(
    const std::string& far_path,
    const std::string& rules,
    const std::string& input_mode,
    const std::string& output_mode) {
  CHECK(grm_.LoadArchive(far_path));
  rules_ = thrax::StringSplit(rules, ',');
  if (rules_.empty()) LOG(FATAL) << "rules must be specified";
  for (size_t i = 0; i < rules_.size(); ++i) {
    thrax::RuleTriple triple(rules_[i]);
    const auto *fst = grm_.GetFst(triple.main_rule);
    if (!fst) {
      LOG(FATAL) << "grm.GetFst() must be non nullptr for rule: "
                 << triple.main_rule;
    }
    fst::StdVectorFst vfst(*fst);
    // If the input transducers in the FAR have symbol tables then we need to
    // add the appropriate symbol table(s) to the input strings, according to
    // the parse mode.
    if (vfst.InputSymbols()) {
      if (!byte_symtab_ &&
          vfst.InputSymbols()->Name() ==
          thrax::function::kByteSymbolTableName) {
        byte_symtab_ = std::shared_ptr<fst::SymbolTable>(
            vfst.InputSymbols()->Copy());
      } else if (!utf8_symtab_ &&
                 vfst.InputSymbols()->Name() ==
                 thrax::function::kUtf8SymbolTableName) {
        utf8_symtab_ = std::shared_ptr<fst::SymbolTable>(
            vfst.InputSymbols()->Copy());
      }
    }
    if (!triple.pdt_parens_rule.empty()) {
      fst = grm_.GetFst(triple.pdt_parens_rule);
      if (!fst) {
        LOG(FATAL) << "grm.GetFst() must be non nullptr for rule: "
                   << triple.pdt_parens_rule;
      }
    }
    if (!triple.mpdt_assignments_rule.empty()) {
      fst = grm_.GetFst(triple.mpdt_assignments_rule);
      if (!fst) {
        LOG(FATAL) << "grm.GetFst() must be non nullptr for rule: "
                   << triple.mpdt_assignments_rule;
      }
    }
  }

  GetGeneratedSymbolTable();
  if (input_mode == "byte") {
    compiler_ = std::shared_ptr<Compiler>(
        new Compiler(fst::StringTokenType::BYTE));
  } else if (input_mode == "utf8") {
    compiler_ = std::shared_ptr<Compiler>(
        new Compiler(fst::StringTokenType::UTF8));
  } else {
    input_symtab_ = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(input_mode));
    if (!input_symtab_) LOG(FATAL) << "Invalid mode or symbol table path.";
    compiler_ = std::shared_ptr<Compiler>(
        new Compiler(fst::StringTokenType::SYMBOL,
                     fst::SymbolTable::ReadText(input_mode)));
  }

  if (output_mode == "byte") {
    type_ = BYTE;
  } else if (output_mode == "utf8") {
    type_ = UTF8;
  } else {
    type_ = SYMBOL;
    output_symtab_ = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(output_mode));
    if (!output_symtab_) LOG(FATAL) << "Invalid mode or symbol table path.";
  }
}

const std::string InverseTextNormalizer::ProcessInput(
    const std::string& input) {
  fst::StdVectorFst input_fst, output_fst;
  if (!(compiler_->operator()(input, &input_fst))) {
    return "Unable to parse input string.";
  }
  FormatFst(&input_fst);
  string return_val = "";
  // Set symbols for the input, if appropriate
  if (byte_symtab_ && type_ == BYTE) {
    input_fst.SetInputSymbols(byte_symtab_.get());
    input_fst.SetOutputSymbols(byte_symtab_.get());
  } else if (utf8_symtab_ && type_ == UTF8) {
    input_fst.SetInputSymbols(utf8_symtab_.get());
    input_fst.SetOutputSymbols(utf8_symtab_.get());
  } else if (input_symtab_ && type_ == SYMBOL) {
    input_fst.SetInputSymbols(input_symtab_.get());
    input_fst.SetOutputSymbols(input_symtab_.get());
  }

  bool succeeded = true;
  for (size_t i = 0; i < rules_.size(); ++i) {
    thrax::RuleTriple triple(rules_[i]);
    if (grm_.Rewrite(triple.main_rule, input_fst, &output_fst,
                     triple.pdt_parens_rule, triple.mpdt_assignments_rule)) {
      input_fst = output_fst;
    } else {
      succeeded = false;
      break;
    }
  }

  std::vector<std::pair<string, float> > strings;
  std::set<string> seen;
  if (succeeded && FstToStrings(output_fst, 1, &strings)) {
    std::vector<std::pair<string, float> >::iterator itr = strings.begin();
    for (; itr != strings.end(); ++itr) {
      std::set<string>::iterator sx = seen.find(itr->first);
      if (sx != seen.end()) continue;
      return_val += itr->first;
      seen.insert(itr->first);
      // for noutput > 1, add space separator
      if (itr + 1 != strings.end()) return_val += " ";
    }
    return return_val;
  } else {
    return "Rewrite failed.";
  }
}

void InverseTextNormalizer::GetGeneratedSymbolTable() {
  const auto* symbolfst = grm_.GetFst("*StringFstSymbolTable");
  if (symbolfst != nullptr) {
    generated_symtab_ = std::shared_ptr<fst::SymbolTable>(
        symbolfst->InputSymbols()->Copy());
  }
}

void InverseTextNormalizer::FormatFst(fst::StdVectorFst *vfst) {
  for (fst::StateIterator<fst::StdVectorFst> state_iter(*vfst);
       !state_iter.Done(); state_iter.Next()) {
    int state_id = state_iter.Value();
    for (fst::MutableArcIterator<fst::StdVectorFst> arc_iter(vfst, state_id);
         !arc_iter.Done(); arc_iter.Next()) {
       const fst::StdArc arc = arc_iter.Value();
       fst::StdArc new_arc(arc.ilabel & 0xff, arc.olabel & 0xff,
                           arc.weight, arc.nextstate);
       arc_iter.SetValue(new_arc);
    }
  }
}

bool InverseTextNormalizer::FstToStrings(
    const fst::StdVectorFst& vfst, const int& n,
    std::vector<std::pair<string, float>>* strings) {
  fst::StdVectorFst shortest_path;
  if (n == 1) {
    fst::ShortestPath(vfst, &shortest_path, n);
  } else {
    // The uniqueness feature of ShortestPath requires us to have an acceptor,
    // so we project and remove epsilon arcs.
    fst::StdVectorFst temp(vfst);
    fst::Project(&temp, fst::PROJECT_OUTPUT);
    fst::RmEpsilon(&temp);
    fst::ShortestPath(temp, &shortest_path, n, /* unique */ true);
  }
  if (shortest_path.Start() == fst::kNoStateId) return false;
  for (fst::PathIterator<fst::StdArc> iter(shortest_path,
                                          /* check_acyclic */ false);
       !iter.Done(); iter.Next()) {
    std::string path;
    for (const auto label : iter.OLabels()) {
      if (!AppendLabel(label, &path)) {
        return false;
      }
    }
    strings->emplace_back(std::move(path), iter.Weight().Value());
  }
  return true;
}

bool InverseTextNormalizer::AppendLabel(const fst::StdArc::Label& label,
                                        std::string* path) {
  if (label != 0) {
    // Check first to see if this label is in the generated symbol set. Note
    // that this should not conflict with a user-provided symbol table since
    // the parser used by GrmCompiler doesn't generate extra labels if a
    // string is parsed using a user-provided symbol table.
    if (generated_symtab_ && !generated_symtab_->Find(label).empty()) {
      string sym = generated_symtab_->Find(label);
      *path += "[" + sym + "]";
    } else if (type_ == SYMBOL) {
      string sym = output_symtab_->Find(label);
      if (sym == "") {
        LOG(ERROR) << "Missing symbol in symbol table for id: " << label;
        return false;
      }
      // For non-byte, non-UTF8 symbols, one overwhelmingly wants these to be
      // space-separated.
      if (!path->empty()) *path += " ";
      *path += sym;
    } else if (type_ == BYTE) {
      path->push_back(label);
    } else if (type_ == UTF8) {
      std::string utf8_string;
      std::vector<fst::StdArc::Label> labels;
      labels.push_back(label);
      if (!fst::LabelsToUTF8String(labels, &utf8_string)) {
        LOG(ERROR) << "LabelsToUTF8String: Bad code point: " << label;
        return false;
      }
      *path += utf8_string;
    }
  }
  return true;
}

}  // namespace wenet
