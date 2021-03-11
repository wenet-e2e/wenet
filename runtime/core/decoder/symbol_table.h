// Copyright (c) 2016 Personal (Binbin Zhang)
// Created on 2016-11-11
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

#ifndef DECODER_SYMBOL_TABLE_H_
#define DECODER_SYMBOL_TABLE_H_

#include <stdio.h>

#include <string>
#include <unordered_map>

#include "glog/logging.h"

namespace wenet {

class SymbolTable {
 public:
  explicit SymbolTable(const std::string& symbol_file) {
    ReadSymbolFile(symbol_file);
  }

  ~SymbolTable() = default;

  const std::string& Find(int id) const { return symbol_tabel_.at(id); }

 private:
  void ReadSymbolFile(const std::string& symbol_file) {
    FILE* fp = fopen(symbol_file.c_str(), "r");
    if (!fp) {
      LOG(FATAL) << symbol_file << " not exist, please check!!!";
    }
    char buffer[1024] = {0}, str[1024] = {0};
    int id;
    while (fgets(buffer, 1024, fp)) {
      int num = sscanf(buffer, "%s %d", str, &id);
      if (num != 2) {
        LOG(FATAL) << "each line shoud have 2 fields, symbol & id";
      }
      CHECK_GE(id, 0);
      std::string symbol = str;
      symbol_tabel_[id] = symbol;
    }
    fclose(fp);
  }

  std::unordered_map<int, std::string> symbol_tabel_;
};

}  // namespace wenet

#endif  // DECODER_SYMBOL_TABLE_H_
