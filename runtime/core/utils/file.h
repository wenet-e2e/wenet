// Copyright (c) 2022  Binbin Zhang (binbzha@qq.com)
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

#ifndef UTILS_FILE_H_
#define UTILS_FILE_H_

#include <fstream>
#include <string>

namespace wenet {

inline bool FileExists(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

}  // namespace wenet

#endif  // UTILS_FILE_H_
