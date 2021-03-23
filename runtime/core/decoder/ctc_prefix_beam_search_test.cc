// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <cmath>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::ElementsAre;

TEST(CtcPrefixBeamSearchTest, CtcPrefixBeamSearchLogicTest) {
  // See https://robin1001.github.io/2020/12/11/ctc-search
  // for the graph demonstration of the data
  std::vector<float> data = {0.25, 0.40, 0.35,
                             0.40, 0.35, 0.25,
                             0.10, 0.50, 0.40};
  torch::Tensor input = torch::from_blob(data.data(), {3, 3}, torch::kFloat);
  ASSERT_THAT(input.sizes(), ElementsAre(3, 3));
  input = torch::log(input);
  wenet::CtcPrefixBeamSearchOptions option;
  option.first_beam_size = 3;
  option.second_beam_size = 3;
  wenet::CtcPrefixBeamSearch beam_search(option);
  beam_search.Search(input);
  // Top N  Path        Timestamp  Score
  //     1  [b:2, a:1]  [0, 2]     0.2185
  //     2  [a:1, b:2]  [0, 2]     0.1550
  //     3  [a:1]       [0]        0.1525
  const std::vector<std::vector<int>>& result = beam_search.hypotheses();
  const std::vector<std::vector<int>>& time_steps = beam_search.time_steps();
  const std::vector<float>& likelihood = beam_search.likelihood();
  EXPECT_EQ(result.size(), 3);
  ASSERT_THAT(result[0], ElementsAre(2, 1));
  ASSERT_THAT(result[1], ElementsAre(1, 2));
  ASSERT_THAT(result[2], ElementsAre(1));

  EXPECT_EQ(time_steps.size(), 3);
  ASSERT_THAT(time_steps[0], ElementsAre(0, 2));
  ASSERT_THAT(time_steps[1], ElementsAre(0, 2));
  ASSERT_THAT(time_steps[2], ElementsAre(0));

  EXPECT_EQ(likelihood.size(), 3);
  EXPECT_FLOAT_EQ(std::exp(likelihood[0]), 0.2185);
  EXPECT_FLOAT_EQ(std::exp(likelihood[1]), 0.1550);
  EXPECT_FLOAT_EQ(std::exp(likelihood[2]), 0.1525);
}
