// Copyright [2021-08-31] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

#include "post_processor/post_processor.h"

#include <vector>
#include <sstream>

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
