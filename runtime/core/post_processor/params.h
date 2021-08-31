// Copyright [2021-08-31] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

#ifndef POST_PROCESSOR_PARAMS_H_
#define POST_PROCESSOR_PARAMS_H_

#include <memory>
#include <vector>

#include "post_processor/post_processor.h"
#include "utils/flags.h"
#include "utils/string.h"

// Space/Blank processor flags
DEFINE_int32(language_type, 0,
             "remove spaces according to language type"
             "0x00 = kMandarinEnglish, "
             "0x01 = kIndoEuropean");
DEFINE_bool(lowercase, true, "lowercase final result if needed");

namespace wenet {
std::shared_ptr<PostProcessOptions> InitPostProcessOptionsFromFlags() {
  auto postprocess_config = std::make_shared<PostProcessOptions>();
  postprocess_config->language_type =
    FLAGS_language_type == 0 ? kMandarinEnglish : kIndoEuropean;
  postprocess_config->lowercase = FLAGS_lowercase;
  return postprocess_config;
}

std::shared_ptr<PostProcessResource> InitPostProcessResourceFromFlags() {
  // TODO(xcsong): add itn related resources
  auto resource = std::make_shared<PostProcessResource>();
  return resource;
}

}  // namespace wenet

#endif  // POST_PROCESSOR_PARAMS_H_
