// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_prefix_beam_search.h"

#include <math.h>
#include <vector>

#include "gtest/gtest.h"

TEST(CtcPrefixBeamSearchTest, CtcPrefixBeamSearchLogicTest) {
  // See https://robin1001.github.io/2020/12/11/ctc-search/ for the
  // graph demonstration of the data
  std::vector<float> data = {0.25, 0.40, 0.35, 0.40, 0.35,
                             0.25, 0.10, 0.50, 0.40};
  torch::Tensor input = torch::from_blob(data.data(), {3, 3}, torch::kFloat);
  EXPECT_EQ(input.size(0), 3);
  EXPECT_EQ(input.size(1), 3);
  input = torch::log(input);
  wenet::CtcPrefixBeamSearchOptions option;
  option.first_beam_size = 3;
  option.second_beam_size = 3;
  wenet::CtcPrefixBeamSearch prefix_beam_search(option);
  prefix_beam_search.Search(input);
  // top 1: [2, 1] 0.2185
  // top 2: [1, 2] 0.1550
  // top 3: [1] 0.1525
  const std::vector<std::vector<int>>& result = prefix_beam_search.hypotheses();
  const std::vector<float>& likelihood = prefix_beam_search.likelihood();
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(likelihood.size(), 3);
  EXPECT_EQ(result[0].size(), 2);
  EXPECT_EQ(result[0][0], 2);
  EXPECT_EQ(result[0][1], 1);
  EXPECT_EQ(result[1].size(), 2);
  EXPECT_EQ(result[1][0], 1);
  EXPECT_EQ(result[1][1], 2);
  EXPECT_EQ(result[2].size(), 1);
  EXPECT_EQ(result[2][0], 1);
  EXPECT_FLOAT_EQ(exp(likelihood[0]), 0.2185);
  EXPECT_FLOAT_EQ(exp(likelihood[1]), 0.1550);
  EXPECT_FLOAT_EQ(exp(likelihood[2]), 0.1525);
}
