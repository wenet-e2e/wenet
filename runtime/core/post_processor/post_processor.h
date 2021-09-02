// Copyright [2021-08-31] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

#ifndef POST_PROCESSOR_POST_PROCESSOR_H_
#define POST_PROCESSOR_POST_PROCESSOR_H_

#include <memory>
#include <string>

#include "utils/utils.h"

namespace wenet {

enum LanguageType {
  // spaces between **mandarin words** should be removed.
  // cases of processing spaces with mandarin-only, english-only
  // and mandarin-english code-switch can be found in post_processor_test.cc
  kMandarinEnglish = 0x00,
  // spaces should be kept for most of the
  // Indo-European languages (i.e., deutsch or english-deutsch code-switch).
  // cases of those languages can be found in post_processor_test.cc
  kIndoEuropean = 0x01
};

struct PostProcessOptions {
  // space options
  // The decoded result may contain spaces (' ' or '_'),
  // we will process those spaces according to language_type. More details can
  // be found in https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
  LanguageType language_type = kMandarinEnglish;
  // whether lowercase letters are required
  bool lowercase = true;
};

// TODO(xcsong): add itn/punctuation related resource
struct PostProcessResource {};

// Post Processor
class PostProcessor {
 public:
  explicit PostProcessor(PostProcessOptions&& opts) : opts_(std::move(opts)) {}
  explicit PostProcessor(const PostProcessOptions& opts) : opts_(opts) {}
  // call other functions to do post processing
  std::string Process(const std::string& str, bool finish);
  // process spaces according to configurations
  std::string ProcessSpace(const std::string& str);
  // TODO(xcsong): add itn/punctuation
  // void InverseTN(const std::string& str);
  // void Punctuate(const std::string& str);

 private:
  const PostProcessOptions opts_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(PostProcessor);
};

}  // namespace wenet

#endif  // POST_PROCESSOR_POST_PROCESSOR_H_
