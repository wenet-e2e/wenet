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
  int bytes = 1;
  for (size_t i = 0; i < str.length(); i += bytes) {
    assert((str[i] & 0xF8) <= 0xF0);
    if ((str[i] & 0x80) == 0x00) {
      // The first 128 characters (US-ASCII) in UTF-8 format only need one byte.
      bytes = 1;
    } else if ((str[i] & 0xE0) == 0xC0) {
      // The next 1,920 characters need two bytes to encode,
      // which covers the remainder of almost all Latin-script alphabets.
      bytes = 2;
    } else if ((str[i] & 0xF0) == 0xE0) {
      // Three bytes are needed for characters in the rest of
      // the Basic Multilingual Plane, which contains virtually all characters
      // in common use, including most Chinese, Japanese and Korean characters.
      bytes = 3;
    } else if ((str[i] & 0xF8) == 0xF0) {
      // Four bytes are needed for characters in the other planes of Unicode,
      // which include less common CJK characters, various historic scripts,
      // mathematical symbols, and emoji (pictographic symbols).
      bytes = 4;
    }
    chars->push_back(str.substr(i, bytes));
  }
}

int UTF8StringLength(const std::string& str) {
  int len = 0;
  int bytes = 1;
  for (size_t i = 0; i < str.length(); i += bytes) {
    if ((str[i] & 0x80) == 0x00) {
      bytes = 1;
    } else if ((str[i] & 0xE0) == 0xC0) {
      bytes = 2;
    } else if ((str[i] & 0xF0) == 0xE0) {
      bytes = 3;
    } else if ((str[i] & 0xF8) == 0xF0) {
      bytes = 4;
    }
    ++len;
  }
  return len;
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

bool SplitUTF8StringToWords(
    const std::string& str,
    const std::shared_ptr<fst::SymbolTable>& symbol_table,
    std::vector<std::string>* words) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(Trim(str), &chars);

  bool no_oov = true;
  for (size_t start = 0; start < chars.size();) {
    for (size_t end = chars.size(); end > start; --end) {
      std::string word;
      for (size_t i = start; i < end; i++) {
        word += chars[i];
      }
      if (symbol_table->Find(word) != -1) {
        words->emplace_back(word);
        start = end;
        continue;
      }
      if (end == start + 1) {
        ++start;
        no_oov = false;
        LOG(WARNING) << word << " is oov.";
      }
    }
  }
  return no_oov;
}

std::string ProcessBlank(const std::string& str, bool lowercase) {
  std::string result;
  if (!str.empty()) {
    std::vector<std::string> chars;
    SplitUTF8StringToChars(Trim(str), &chars);

    for (std::string& ch : chars) {
      if (ch != kSpaceSymbol) {
        result.append(ch);
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
    // NOTE: convert string to wstring
    //       see issue 745: https://github.com/wenet-e2e/wenet/issues/745
    std::locale loc("");
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring wsresult = converter.from_bytes(result);
    for (auto& c : wsresult) {
      c = lowercase ? tolower(c, loc) : toupper(c, loc);
    }
    result = converter.to_bytes(wsresult);
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

std::string Trim(const std::string& str) { return Rtrim(Ltrim(str)); }

std::string JoinPath(const std::string& left, const std::string& right) {
  std::string path(left);
  if (path.size() && path.back() != '/') {
    path.push_back('/');
  }
  path.append(right);
  return path;
}

#ifdef _MSC_VER
std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}
#endif

}  // namespace wenet
