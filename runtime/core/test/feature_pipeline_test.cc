// Copyright (c) 2022 Roney
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frontend/feature_pipeline.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(FeaturePipelineTest, PipelineTest) {
  wenet::FeaturePipelineConfig config(80, 8000);
  wenet::FeaturePipeline feature_pipeline(config);
  int audio_len = 8 * 55;  // audio len 55ms,4 frames
  std::vector<float> pcm(audio_len, 0);
  feature_pipeline.AcceptWaveform(pcm.data(), audio_len);
  ASSERT_EQ(feature_pipeline.NumQueuedFrames(), 4);

  std::vector<std::vector<float>> out_feats;
  auto b = feature_pipeline.Read(2, &out_feats);
  ASSERT_TRUE(b);
  ASSERT_EQ(out_feats.size(), 2);
  ASSERT_EQ(feature_pipeline.NumQueuedFrames(), 2);

  std::vector<float> out_feat;
  b = feature_pipeline.ReadOne(&out_feat);
  ASSERT_TRUE(b);
  ASSERT_FALSE(out_feat.empty());
  ASSERT_EQ(feature_pipeline.NumQueuedFrames(), 1);

  feature_pipeline.set_input_finished();
  b = feature_pipeline.Read(2, &out_feats);
  ASSERT_FALSE(b);
  ASSERT_EQ(out_feats.size(), 1);
  ASSERT_EQ(feature_pipeline.NumQueuedFrames(), 0);

  feature_pipeline.AcceptWaveform(pcm.data(), audio_len);
  feature_pipeline.Read(2, &out_feats);
  feature_pipeline.Reset();
  feature_pipeline.set_input_finished();
  b = feature_pipeline.Read(2, &out_feats);
  ASSERT_FALSE(b);
  ASSERT_EQ(out_feats.size(), 0);
  ASSERT_EQ(feature_pipeline.NumQueuedFrames(), 0);
}
