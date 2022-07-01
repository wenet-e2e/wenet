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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "utils/utils.h"

TEST(PostProcessorTest, ProcessSpacekMandarinEnglishTest) {
  wenet::PostProcessOptions opts_lowercase;
  wenet::PostProcessor post_processor_lowercase(opts_lowercase);

  wenet::PostProcessOptions opts_uppercase;
  opts_uppercase.lowercase = false;
  wenet::PostProcessor post_processor_uppercase(opts_uppercase);

  std::vector<std::string> input = {
      // modeling unit: mandarin character
      // decode type: CtcPrefixBeamSearch, "".join()
      "震东好帅",
      // modeling unit: mandarin word
      // decode type: CtcWfstBeamSearch, " ".join()
      " 吴迪 也 好帅",
      // modeling unit: english wordpiece
      // decode type: CtcPrefixBeamSearch, "".join()
      "▁binbin▁is▁also▁handsome",
      // modeling unit: english word
      // decode type: CtcWfstBeamSearch, " ".join()
      " life is short i use wenet",
      // modeling unit: mandarin character + english wordpiece
      // decode type: CtcPrefixBeamSearch, "".join()
      "超哥▁is▁the▁most▁handsome",
      // modeling unit: mandarin word + english word
      // decode type: CtcWfstBeamSearch, " ".join()
      " 人生 苦短 i use wenet",
  };

  std::vector<std::string> result_lowercase = {
      "震东好帅",
      "吴迪也好帅",
      "binbin is also handsome",
      "life is short i use wenet",
      "超哥 is the most handsome",
      "人生苦短i use wenet",
  };

  std::vector<std::string> result_uppercase = {
      "震东好帅",
      "吴迪也好帅",
      "BINBIN IS ALSO HANDSOME",
      "LIFE IS SHORT I USE WENET",
      "超哥 IS THE MOST HANDSOME",
      "人生苦短I USE WENET",
  };

  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(post_processor_lowercase.ProcessSpace(input[i]),
              result_lowercase[i]);
    EXPECT_EQ(post_processor_uppercase.ProcessSpace(input[i]),
              result_uppercase[i]);
  }
}

TEST(PostProcessorTest, ProcessSpacekIndoEuropeanTest) {
  wenet::PostProcessOptions opts_lowercase;
  opts_lowercase.language_type = wenet::kIndoEuropean;
  wenet::PostProcessor post_processor_lowercase(opts_lowercase);

  wenet::PostProcessOptions opts_uppercase;
  opts_uppercase.language_type = wenet::kIndoEuropean;
  opts_uppercase.lowercase = false;
  wenet::PostProcessor post_processor_uppercase(opts_uppercase);

  std::vector<std::string> input = {
      // modeling unit: wordpiece
      // decode type: CtcPrefixBeamSearch, "".join()
      "▁zhendong▁ist▁so▁schön",
      // modeling unit: word
      // decode type: CtcWfstBeamSearch, " ".join()
      " zhendong ist so schön"};

  std::vector<std::string> result_lowercase = {"zhendong ist so schön",
                                               "zhendong ist so schön"};

  std::vector<std::string> result_uppercase = {"ZHENDONG IST SO SCHÖN",
                                               "ZHENDONG IST SO SCHÖN"};

  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(post_processor_lowercase.ProcessSpace(input[i]),
              result_lowercase[i]);
    EXPECT_EQ(post_processor_uppercase.ProcessSpace(input[i]),
              result_uppercase[i]);
  }
}
