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

#ifndef UTILS_STRING_H_
#define UTILS_STRING_H_

#include <string>
#include <vector>

namespace wenet {

void SplitString(const std::string& str, std::vector<std::string>* strs);

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out);

bool SplitUTF8String(const std::string& str,
                     std::vector<std::string>* characters);

// Remove head,tail and consecutive space.
std::string ProcessBlank(const std::string& str);

// NOTE(Xingchen Song): we add this function to make it possible to
// support multilingual recipe in the future, in which characters of
// different languages are all encoded in UTF-8 format.
// UTF-8 REF: https://en.wikipedia.org/wiki/UTF-8#Encoding
void SplitEachChar(const std::string& word, std::vector<std::string>* chars);

bool CheckEnglishWord(const std::string& word);

}  // namespace wenet

#endif  // UTILS_STRING_H_
