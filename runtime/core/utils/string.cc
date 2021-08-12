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

#include "utils/string.h"

#include <sstream>
#include <string>
#include <vector>

#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

void SplitString(const std::string& str, std::vector<std::string>* strs) {
  SplitStringToVector(Trim(str), " \t", true, strs);
}

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

void SplitUTF8StringToChars(const std::string& str,
                            std::vector<std::string>* chars) {
  chars->clear();
  size_t i = 0;
  while (i < str.length()) {
    assert((str[i] & 0xF8) <= 0xF0);
    int bytes_ = 1;
    if ((str[i] & 0x80) == 0x00) {
      // The first 128 characters (US-ASCII) in UTF-8 format only need one byte.
      bytes_ = 1;
    } else if ((str[i] & 0xE0) == 0xC0) {
      // The next 1,920 characters need two bytes to encode,
      // which covers the remainder of almost all Latin-script alphabets.
      bytes_ = 2;
    } else if ((str[i] & 0xF0) == 0xE0) {
      // Three bytes are needed for characters in the rest of
      // the Basic Multilingual Plane, which contains virtually all characters
      // in common use, including most Chinese, Japanese and Korean characters.
      bytes_ = 3;
    } else if ((str[i] & 0xF8) == 0xF0) {
      // Four bytes are needed for characters in the other planes of Unicode,
      // which include less common CJK characters, various historic scripts,
      // mathematical symbols, and emoji (pictographic symbols).
      bytes_ = 4;
    }
    chars->push_back(str.substr(i, bytes_));
    i += bytes_;
  }
}

bool CheckEnglishChar(const std::string& ch) {
  // all english characters should be encoded in one byte
  if (ch.size() != 1) return false;
  // english words may contain apostrophe, i.e., "He's"
  return isalpha(ch[0]) || ch[0] == '\'';
}

bool CheckEnglishWord(const std::string& word) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(word, &chars);
  for (size_t k = 0; k < chars.size(); k++) {
    if (!CheckEnglishChar(chars[k])) {
      return false;
    }
  }
  return true;
}

std::string JoinString(const std::string& c,
                       const std::vector<std::string>& strs) {
  std::string result;
  if (strs.size() > 0) {
    for (int i = 0; i < strs.size() - 1; i++) {
      result += (strs[i] + c);
    }
    result += strs.back();
  }
  return result;
}

void SplitUTF8StringToWords(const std::string& str,
                            std::vector<std::string>* words) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(Trim(str), &chars);

  words->clear();
  std::ostringstream oss;
  // concat english chars into word
  bool is_english_current = false;
  for (const std::string& ch : chars) {
    if (ch.empty() || ch[0] == ' ') continue;
    if (CheckEnglishChar(ch)) {
      is_english_current = true;
      oss << ch;
    } else {
      // push back the complete english word to words
      if (is_english_current) {
        is_english_current = false;
        words->push_back(oss.str());
        oss.str("");
      }
      words->push_back(ch);
    }
  }
  // push back the last english word to words
  if (is_english_current) {
    words->push_back(oss.str());
  }
}

std::string ProcessBlank(const std::string& str) {
  std::string result;
  if (!str.empty()) {
    std::vector<std::string> characters;
    SplitUTF8StringToChars(Trim(str), &characters);

    for (std::string& character : characters) {
      if (character != kSpaceSymbol) {
        result.append(character);
      } else {
        // Ignore consecutive space or located in head
        if (!result.empty() && result.back() != ' ') {
          result.push_back(' ');
        }
      }
    }
    // Ignore tailing space
    if (!result.empty() && result.back() == ' ') {
      result.pop_back();
    }
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = tolower(result[i]);
    }
  }
  return result;
}

std::string Ltrim(const std::string& str) {
  size_t start = str.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : str.substr(start);
}

std::string Rtrim(const std::string& str) {
  size_t end = str.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

std::string Trim(const std::string& str) {
  return Rtrim(Ltrim(str));
}

}  // namespace wenet
