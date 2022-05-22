// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbzha@qq.com (Binbin Zhang)

#include "utils/utils.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(UtilsTest, TopKTest) {
  using ::testing::ElementsAre;
  using ::testing::FloatNear;
  using ::testing::Pointwise;
  std::vector<float> data = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  std::vector<float> values;
  std::vector<int32_t> indices;
  wenet::TopK(data, 3, &values, &indices);
  EXPECT_THAT(values, Pointwise(FloatNear(1e-8), {10, 9, 8}));
  ASSERT_THAT(indices, ElementsAre(9, 4, 8));
}
