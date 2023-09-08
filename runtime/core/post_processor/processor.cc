// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "post_processor/processor.h"

using fst::StringTokenType;

namespace wenet {
Processor::Processor(const std::string& tagger_path,
                     const std::string& verbalizer_path) {
  tagger_.reset(StdVectorFst::Read(tagger_path));
  verbalizer_.reset(StdVectorFst::Read(verbalizer_path));
  compiler_ = std::make_shared<StringCompiler<StdArc>>(StringTokenType::BYTE);
  printer_ = std::make_shared<StringPrinter<StdArc>>(StringTokenType::BYTE);

  if (tagger_path.find("_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kTN;
  } else if (tagger_path.find("_itn_") != tagger_path.npos) {
    parse_type_ = ParseType::kITN;
  } else {
    LOG(FATAL) << "Invalid fst prefix, prefix should contain"
               << " either \"_tn_\" or \"_itn_\".";
  }
}

std::string Processor::shortest_path(const StdVectorFst& lattice) {
  StdVectorFst shortest_path;
  fst::ShortestPath(lattice, &shortest_path, 1, true);

  std::string output;
  printer_->operator()(shortest_path, &output);
  return output;
}

std::string Processor::compose(const std::string& input,
                               const StdVectorFst* fst) {
  StdVectorFst input_fst;
  compiler_->operator()(input, &input_fst);

  StdVectorFst lattice;
  fst::Compose(input_fst, *fst, &lattice);
  return shortest_path(lattice);
}

std::string Processor::tag(const std::string& input) {
  return compose(input, tagger_.get());
}

std::string Processor::verbalize(const std::string& input) {
  if (input.empty()) {
    return "";
  }
  TokenParser parser(parse_type_);
  std::string output = parser.reorder(input);
  return compose(output, verbalizer_.get());
}

std::string Processor::normalize(const std::string& input) {
  return verbalize(tag(input));
}

}  // namespace wenet
