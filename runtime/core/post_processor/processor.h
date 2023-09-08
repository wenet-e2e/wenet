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

#ifndef POST_PROCESSOR_PROCESSOR_H_
#define POST_PROCESSOR_PROCESSOR_H_

#include "fst/fstlib.h"

#include "post_processor/token_parser.h"

using fst::StdArc;
using fst::StdVectorFst;
using fst::StringCompiler;
using fst::StringPrinter;

namespace wenet {
class Processor {
 public:
  Processor(const std::string& tagger_path, const std::string& verbalizer_path);
  std::string tag(const std::string& input);
  std::string verbalize(const std::string& input);
  std::string normalize(const std::string& input);

 private:
  std::string shortest_path(const StdVectorFst& lattice);
  std::string compose(const std::string& input, const StdVectorFst* fst);

  ParseType parse_type_;
  std::shared_ptr<StdVectorFst> tagger_ = nullptr;
  std::shared_ptr<StdVectorFst> verbalizer_ = nullptr;
  std::shared_ptr<StringCompiler<StdArc>> compiler_ = nullptr;
  std::shared_ptr<StringPrinter<StdArc>> printer_ = nullptr;
};

}  // namespace wenet

#endif  // POST_PROCESSOR_PROCESSOR_H_
