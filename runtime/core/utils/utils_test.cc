// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbzha@qq.com (Binbin Zhang)

#include "utils/utils.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"


TEST(TopKTest, TopKLogicTest) {
  using ::testing::ElementsAre;
  std::vector<int> data = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  std::vector<int> values;
  std::vector<int32_t> indices;
  wenet::TopK(data, 3, &values, &indices);
  EXPECT_EQ(values.size(), 3);
  EXPECT_EQ(indices.size(), 3);
  ASSERT_THAT(values, ElementsAre(10, 9, 8));
  ASSERT_THAT(indices, ElementsAre(9, 4, 8))
}
