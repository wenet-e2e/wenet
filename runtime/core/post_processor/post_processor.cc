// Copyright (c) 2021 Xingchen Song sxc19@mails.tsinghua.edu.cn
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#include "post_processor/post_processor.h"

#include <sstream>
#include <vector>

#include "utils/string.h"

namespace wenet {

std::string PostProcessor::ProcessSpace(const std::string& str) {
  std::string result = str;
  // 1. remove ' ' if needed
  // only spaces between mandarin words need to be removed, please note that
  // if str contains '_', we assume that the decoding type must be
  // `CtcPrefixBeamSearch` and this branch will do nothing since str must be
  // obtained via "".join() (in function `TorchAsrDecoder::UpdateResult()`)
  if (opts_.language_type == kMandarinEnglish && !str.empty()) {
    result.clear();
    // split str by ' '
    std::vector<std::string> words;
    std::stringstream ss(str);
    std::string tmp;
    while (ss >> tmp) {
      words.push_back(tmp);
    }
    // check english word
    bool is_englishword_prev = false;
    bool is_englishword_now = false;
    for (std::string& w : words) {
      is_englishword_now = CheckEnglishWord(w);
      if (is_englishword_prev && is_englishword_now) {
        result += (' ' + w);
      } else {
        result += (w);
      }
      is_englishword_prev = is_englishword_now;
    }
  }
  // 2. replace '_' with ' '
  // this should be done for all cases (both kMandarinEnglish and kIndoEuropean)
  result = ProcessBlank(result, opts_.lowercase);
  return result;
}

std::string PostProcessor::Process(const std::string& str, bool finish) {
  std::string result;
  result = ProcessSpace(str);
  // TODO(xcsong): do itn/punctuation if finish == true
  return result;
}

}  // namespace wenet
