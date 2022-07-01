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

#include <codecvt>
#include <locale>
#include <memory>
#include <string>
#include <vector>

#include "fst/symbol-table.h"

namespace wenet {

const char WHITESPACE[] = " \n\r\t\f\v";

// Split the string with space or tab.
void SplitString(const std::string& str, std::vector<std::string>* strs);

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out);

// NOTE(Xingchen Song): we add this function to make it possible to
// support multilingual recipe in the future, in which characters of
// different languages are all encoded in UTF-8 format.
// UTF-8 REF: https://en.wikipedia.org/wiki/UTF-8#Encoding
// Split the UTF-8 string into chars.
void SplitUTF8StringToChars(const std::string& str,
                            std::vector<std::string>* chars);

int UTF8StringLength(const std::string& str);

// Check whether the UTF-8 char is alphabet or '.
bool CheckEnglishChar(const std::string& ch);

// Check whether the UTF-8 word is only contains alphabet or '.
bool CheckEnglishWord(const std::string& word);

std::string JoinString(const std::string& c,
                       const std::vector<std::string>& strs);

// Split the UTF-8 string into words by symbol table.
// Return whether not contains oov.
bool SplitUTF8StringToWords(
    const std::string& str,
    const std::shared_ptr<fst::SymbolTable>& symbol_table,
    std::vector<std::string>* words);

// Replace ▁ with space, then remove head, tail and consecutive space.
std::string ProcessBlank(const std::string& str, bool lowercase);

std::string Ltrim(const std::string& str);

std::string Rtrim(const std::string& str);

std::string Trim(const std::string& str);

std::string JoinPath(const std::string& left, const std::string& right);

#ifdef _MSC_VER
std::wstring ToWString(const std::string& str);
#endif

}  // namespace wenet

#endif  // UTILS_STRING_H_
