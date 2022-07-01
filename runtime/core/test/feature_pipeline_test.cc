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

#include <thread>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "utils/blocking_queue.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

void pushQueue(const std::shared_ptr<wenet::BlockingQueue<int>>& que,
               std::vector<int> vec) {
  que->Push(vec);
}

void popQueue(const std::shared_ptr<wenet::BlockingQueue<int>>& que, int num,
              int back_data) {
  auto pop_data = que->Pop(num);
  ASSERT_EQ(pop_data[num - 1], back_data);
}

TEST(FeaturePipelineTest, BlockingQueueTest) {
  auto capacity_queue = std::make_shared<wenet::BlockingQueue<int>>(2);
  std::vector<int> test_data{1, 2, 3, 4, 5};
  std::thread push_thread(&pushQueue, capacity_queue, test_data);
  ASSERT_EQ(capacity_queue->Pop(), 1);
  ASSERT_LE(capacity_queue->Size(), 2);    // capacity_queue: 2 or 2,3
  auto pop_data = capacity_queue->Pop(3);  // 2,3,4 num > capacity
  ASSERT_EQ(pop_data.size(), 3);
  ASSERT_EQ(pop_data[2], 4);
  push_thread.join();
  ASSERT_EQ(capacity_queue->Size(), 1);  // capacity_queue:5

  std::thread pop_thread(&popQueue, capacity_queue, 3, 0);  // num > capacity
  capacity_queue->Push(9);  // capacity_queue:5,9
  capacity_queue->Push(0);  // capacity_queue:5,9,0
  pop_thread.join();        // capacity_queue:
  ASSERT_EQ(capacity_queue->Size(), 0);

  pop_data = capacity_queue->Pop(0);
  ASSERT_TRUE(pop_data.empty());
}

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
