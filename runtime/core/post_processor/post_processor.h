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

#ifndef POST_PROCESSOR_POST_PROCESSOR_H_
#define POST_PROCESSOR_POST_PROCESSOR_H_

#include <memory>
#include <string>
#include <utility>

#ifdef USE_ITN
#include "text_processor/text_processor.h"
#endif

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
  // be found in
  // https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
  LanguageType language_type = kMandarinEnglish;
  // whether lowercase letters are required
  bool lowercase = true;
};

// TODO(xcsong): add punctuation related resource
struct PostProcessResource {
#ifdef USE_ITN
  std::shared_ptr<TextProcessor> itn_processor = nullptr;
#endif
};

// Post Processor
class PostProcessor {
 public:
  explicit PostProcessor(PostProcessOptions&& opts) : opts_(std::move(opts)) {}
  explicit PostProcessor(const PostProcessOptions& opts) : opts_(opts) {}
#ifdef USE_ITN
  explicit PostProcessor(PostProcessOptions&& opts,
                         std::shared_ptr<PostProcessResource> resource)
    : opts_(std::move(opts)), itn_processor_(resource->itn_processor) {}
  explicit PostProcessor(const PostProcessOptions& opts,
                         std::shared_ptr<PostProcessResource> resource)
    : opts_(opts), itn_processor_(resource->itn_processor) {}
#endif
  // call other functions to do post processing
  std::string Process(const std::string& str, bool finish);
  // process spaces according to configurations
  std::string ProcessSpace(const std::string& str);

 private:
  const PostProcessOptions opts_;
#ifdef USE_ITN
  std::shared_ptr<TextProcessor> itn_processor_ = nullptr;
#endif

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(PostProcessor);
};

}  // namespace wenet

#endif  // POST_PROCESSOR_POST_PROCESSOR_H_
