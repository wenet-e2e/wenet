// Copyright (c) 2021.
// Author: sxc19@mails.tsinghua.edu.cn (Xingchen Song)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "backend/inverse_text_normalizer.h"
#include "backend/params.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(InverseTextNormalizeTest, TestCases53) {
  const std::string far_path = "itn.far";
  std::unique_ptr<wenet::InverseTextNormalizer> model(
      new wenet::InverseTextNormalizer());
  model->Initialize(far_path, FLAGS_rules, FLAGS_input_mode, FLAGS_output_mode);
  std::string text, result;

  // case 1
  text = "一";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "1");

  // case 2
  text = "二";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "2");

  // case 3
  text = "三";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "3");

  // case 4
  text = "四";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "4");

  // case 5
  text = "五";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "5");

  // case 6
  text = "六";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "6");

  // case 7
  text = "七";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "7");

  // case 8
  text = "八";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "8");

  // case 9
  text = "九";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "9");

  // case 10
  text = "十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "10");

  // case 11
  text = "十九";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "19");

  // case 12
  text = "二十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "20");

  // case 13
  text = "二十八";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "28");

  // case 14
  text = "三十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "30");

  // case 15
  text = "三十七";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "37");

  // case 16
  text = "四十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "40");

  // case 17
  text = "四十六";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "46");

  // case 18
  text = "五十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "50");

  // case 19
  text = "五十五";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "55");

  // case 20
  text = "六十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "60");

  // case 21
  text = "六十四";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "64");

  // case 22
  text = "七十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "70");

  // case 23
  text = "七十三";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "73");

  // case 24
  text = "八十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "80");

  // case 25
  text = "八十二";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "82");

  // case 26
  text = "九十";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "90");

  // case 27
  text = "九十一";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "91");

  // case 28
  text = "一百";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "100");

  // case 29
  text = "一百零一";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "101");

  // case 30
  text = "一百一";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "110");

  // case 31
  text = "一百一十二";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "112");

  // case 32
  text = "两百二";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "220");

  // case 33
  text = "二百二十三";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "223");

  // case 34
  text = "三百三";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "330");

  // case 35
  text = "三百三十四";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "334");

  // case 36
  text = "四百四";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "440");

  // case 37
  text = "四百四十五";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "445");

  // case 38
  text = "一千五百五";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "1550");

  // case 39
  text = "两千五百五十六";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "2556");

  // case 40
  text = "三千六百六";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "3660");

  // case 41
  text = "四千六百六十七";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "4667");

  // case 42
  text = "五千七百七";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "5770");

  // case 43
  text = "六千七百七十八";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "6778");

  // case 44
  text = "七千八百八";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "7880");

  // case 45
  text = "八千八百八十九";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "8889");

  // case 46
  text = "九千九百九";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "9990");

  // case 47
  text = "九千九百九十一";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "9991");

  // case 48
  text = "二零一九年九月十二日";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "2019年9月12日");

  // case 49
  text = "两千零五年八月五号";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "2005年8月5号");

  // case 50
  text = "八五年二月二十七日";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "85年2月27日");

  // case 51
  text = "公元一六三年";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "公元163年");

  // case 52
  text = "零六年一月二号";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "06年1月2号");

  // case 53
  text = "有百分之六十二的人认为";
  result = model->ProcessInput(text);
  EXPECT_EQ(result, "有62%的人认为");
}

