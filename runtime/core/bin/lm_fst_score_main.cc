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

#include <iostream>
#include <string>
#include <vector>

#include "lm/lm_fst.h"
#include "utils/flags.h"
#include "utils/log.h"
#include "utils/utils.h"

DEFINE_string(lm_fst_path, "", "arpa fst lm file");
DEFINE_string(dict_path, "", "dict path");

int main(int argc, char *argv[]) {
  ParseCommandLineFlags(&argc, &argv, false);
  LOG(INFO) << FLAGS_lm_fst_path;
  LOG(INFO) << FLAGS_dict_path;
  auto symbol_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_dict_path));
  wenet::LmFst lm_fst(FLAGS_lm_fst_path, symbol_table);

  while (true) {
    std::string line;
    std::getline(std::cin, line);
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    lm_fst.StepTokenArray(strs);
  }
  return 0;
}
