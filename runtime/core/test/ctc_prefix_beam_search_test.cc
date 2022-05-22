// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <cmath>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "utils/utils.h"

TEST(CtcPrefixBeamSearchTest, CtcPrefixBeamSearchLogicTest) {
  using ::testing::ElementsAre;
  // See https://robin1001.github.io/2020/12/11/ctc-search for the
  // graph demonstration of the data
  std::vector<std::vector<float>> data = {
      {0.25, 0.40, 0.35}, {0.40, 0.35, 0.25}, {0.10, 0.50, 0.40}};
  // Apply log
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size(); j++) {
      data[i][j] = std::log(data[i][j]);
    }
  }
  wenet::CtcPrefixBeamSearchOptions option;
  option.first_beam_size = 3;
  option.second_beam_size = 3;
  wenet::CtcPrefixBeamSearch prefix_beam_search(option);
  prefix_beam_search.Search(data);
  /* Test case info
  | top k | result index | prefix score | viterbi score | timestamp |
  |-------|--------------|--------------|---------------|-----------|
  | top 1 | [2, 1]       | 0.2185       | 0.07          | [0, 2]    |
  | top 2 | [1, 2]       | 0.1550       | 0.064         | [0, 2]    |
  | top 3 | [1]          | 0.1525       | 0.07          | [2]       |
  */
  const std::vector<std::vector<int>>& result = prefix_beam_search.Outputs();
  EXPECT_EQ(result.size(), 3);
  ASSERT_THAT(result[0], ElementsAre(2, 1));
  ASSERT_THAT(result[1], ElementsAre(1, 2));
  ASSERT_THAT(result[2], ElementsAre(1));

  const std::vector<float>& likelihood = prefix_beam_search.Likelihood();
  EXPECT_EQ(likelihood.size(), 3);
  EXPECT_FLOAT_EQ(std::exp(likelihood[0]), 0.2185);
  EXPECT_FLOAT_EQ(std::exp(likelihood[1]), 0.1550);
  EXPECT_FLOAT_EQ(std::exp(likelihood[2]), 0.1525);

  const std::vector<float>& viterbi_likelihood =
      prefix_beam_search.viterbi_likelihood();
  EXPECT_EQ(viterbi_likelihood.size(), 3);
  EXPECT_FLOAT_EQ(std::exp(viterbi_likelihood[0]), 0.07);
  EXPECT_FLOAT_EQ(std::exp(viterbi_likelihood[1]), 0.064);
  EXPECT_FLOAT_EQ(std::exp(viterbi_likelihood[2]), 0.07);

  const std::vector<std::vector<int>>& times = prefix_beam_search.Times();
  EXPECT_EQ(times.size(), 3);
  ASSERT_THAT(times[0], ElementsAre(0, 2));
  ASSERT_THAT(times[1], ElementsAre(0, 2));
  ASSERT_THAT(times[2], ElementsAre(2));
}
