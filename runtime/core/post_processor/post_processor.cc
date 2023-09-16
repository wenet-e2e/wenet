// Copyright (c) 2021 Xingchen Song sxc19@mails.tsinghua.edu.cn
//               2023 Jing Du (thuduj12@163.com)
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
#include "processor/wetext_processor.h"
#include "utils/string.h"

namespace wenet {
void PostProcessor::InitITNResource(const std::string& tagger_path,
                                    const std::string& verbalizer_path) {
  auto itn_processor =
      std::make_shared<wetext::Processor>(tagger_path, verbalizer_path);
  itn_resource = itn_processor;
}

std::string PostProcessor::ProcessSpace(const std::string& str) {
  std::string result = str;
  // 1. remove ' ' if needed
  // only spaces between mandarin words need to be removed, please note that
  // if str contains '_', we assume that the decoding type must be
  // `CtcPrefixBeamSearch` and this branch will do nothing since str must be
  // obtained via "".join() (in function `AsrDecoder::UpdateResult()`)
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

std::string del_substr(const std::string& str, const std::string& sub) {
  std::string result = str;
  int pos = 0;
  while (string::npos != (pos = result.find(sub))) {
    result.erase(pos, sub.size());
  }
  return result;
}

std::string PostProcessor::ProcessSymbols(const std::string& str) {
  std::string result = str;
  result = del_substr(result, "<unk>");
  result = del_substr(result, "<context>");
  result = del_substr(result, "</context>");
  return result;
}

std::string PostProcessor::Process(const std::string& str, bool finish) {
  std::string result;
  // remove symbols with "<>" first
  result = ProcessSymbols(str);
  result = ProcessSpace(result);
  // TODO(xcsong): do punctuation if finish == true
  if (finish == true && opts_.itn) {
    if (nullptr != itn_resource) {
      result = itn_resource->Normalize(result);
    }
  }
  return result;
}

}  // namespace wenet
