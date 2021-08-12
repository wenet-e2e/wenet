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
  SplitStringToVector(str, " \t", true, strs);
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

std::string UTF8CodeToUTF8String(int code) {
  std::ostringstream ostr;
  if (code < 0) {
    LOG(ERROR) << "LabelsToUTF8String: Invalid character found: " << code;
    return ostr.str();
  } else if (code < 0x80) {
    ostr << static_cast<char>(code);
  } else if (code < 0x800) {
    ostr << static_cast<char>((code >> 6) | 0xc0);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x10000) {
    ostr << static_cast<char>((code >> 12) | 0xe0);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x200000) {
    ostr << static_cast<char>((code >> 18) | 0xf0);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x4000000) {
    ostr << static_cast<char>((code >> 24) | 0xf8);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else {
    ostr << static_cast<char>((code >> 30) | 0xfc);
    ostr << static_cast<char>(((code >> 24) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  }
  return ostr.str();
}

// Split utf8 string into characters.
bool SplitUTF8String(const std::string& str,
                     std::vector<std::string>* characters) {
  const char* data = str.data();
  const size_t length = str.size();
  for (size_t i = 0; i < length; /* no update */) {
    int c = data[i++] & 0xff;
    if ((c & 0x80) == 0) {
      characters->push_back(UTF8CodeToUTF8String(c));
    } else {
      if ((c & 0xc0) == 0x80) {
        LOG(ERROR) << "UTF8StringToLabels: continuation byte as lead byte";
        return false;
      }
      int count =
          (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) + (c >= 0xfc);
      int code = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (i == length) {
          LOG(ERROR) << "UTF8StringToLabels: truncated utf-8 byte sequence";
          return false;
        }
        char cb = data[i++];
        if ((cb & 0xc0) != 0x80) {
          LOG(ERROR) << "UTF8StringToLabels: missing/invalid continuation byte";
          return false;
        }
        code = (code << 6) | (cb & 0x3f);
        count--;
      }
      if (code < 0) {
        // This should not be able to happen.
        LOG(ERROR) << "UTF8StringToLabels: Invalid character found: " << c;
        return false;
      }
      characters->push_back(UTF8CodeToUTF8String(code));
    }
  }
  return true;
}

std::string ProcessBlank(const std::string& str) {
  std::string result;
  if (!str.empty()) {
    std::vector<std::string> characters;
    if (SplitUTF8String(str, &characters)) {
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
  }
  return result;
}

void SplitEachChar(const std::string& word, std::vector<std::string>* chars) {
  chars->clear();
  size_t i = 0;
  while (i < word.length()) {
    assert((word[i] & 0xF8) <= 0xF0);
    int bytes_ = 1;
    if ((word[i] & 0x80) == 0x00) {
      // The first 128 characters (US-ASCII) in UTF-8 format only need one byte.
      bytes_ = 1;
    } else if ((word[i] & 0xE0) == 0xC0) {
      // The next 1,920 characters need two bytes to encode,
      // which covers the remainder of almost all Latin-script alphabets.
      bytes_ = 2;
    } else if ((word[i] & 0xF0) == 0xE0) {
      // Three bytes are needed for characters in the rest of
      // the Basic Multilingual Plane, which contains virtually all characters
      // in common use, including most Chinese, Japanese and Korean characters.
      bytes_ = 3;
    } else if ((word[i] & 0xF8) == 0xF0) {
      // Four bytes are needed for characters in the other planes of Unicode,
      // which include less common CJK characters, various historic scripts,
      // mathematical symbols, and emoji (pictographic symbols).
      bytes_ = 4;
    }
    chars->push_back(word.substr(i, bytes_));
    i += bytes_;
  }
  return;
}

bool CheckEnglishWord(const std::string& word) {
  std::vector<std::string> chars;
  SplitEachChar(word, &chars);
  for (size_t k = 0; k < chars.size(); k++) {
    // all english characters should be encoded in one byte
    if (chars[k].size() > 1) return false;
    // english words may contain apostrophe, i.e., "He's"
    if (chars[k][0] == '\'') continue;
    if (!isalpha(chars[k][0])) return false;
  }
  return true;
}

}  // namespace wenet
