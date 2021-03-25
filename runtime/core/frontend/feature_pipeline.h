// Copyright (c) 2017 Personal (Binbin Zhang)
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

#ifndef FRONTEND_FEATURE_PIPELINE_H_
#define FRONTEND_FEATURE_PIPELINE_H_

#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "frontend/fbank.h"
#include "utils/blocking_queue.h"

namespace wenet {

struct FeaturePipelineConfig {
  int num_bins;
  int sample_rate;
  int frame_length;
  int frame_shift;
  FeaturePipelineConfig()
      : num_bins(80),        // 80 dim fbank
        sample_rate(16000),  // 16k sample rate
        frame_length(400),   // frame length 25ms
        frame_shift(160) {   // frame shift 10ms
  }

  void Info() const {
    LOG(INFO) << "feature pipeline config"
              << " num_bins " << num_bins << " frame_length " << frame_length
              << "frame_shift" << frame_shift;
  }
};

// Typically, FeaturePipeline is used in two threads: one thread call
// AcceptWaveform() to add raw wav data, another thread call Read() to read
// feature. so it is important to make it thread safe, and the Read() call
// should be a blocking call when there is no feature in feature_queue_ while
// the input is not finished.

class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);

  void AcceptWaveform(const std::vector<float>& wav);
  int num_frames() const { return num_frames_; }
  int feature_dim() const { return feature_dim_; }
  const FeaturePipelineConfig& config() const { return config_; }
  void set_input_finished();

  // Return false if input_finished_ and there is no feature left in
  // feature_queue_
  bool ReadOne(std::vector<float>* feat);
  // Return value is the same to ReadOne
  bool Read(int num_frames, std::vector<std::vector<float>>* feats);

  void Reset();
  bool IsLastFrame(int frame) const {
    return input_finished_ && (frame == num_frames_ - 1);
  }

 private:
  const FeaturePipelineConfig& config_;
  int feature_dim_;
  Fbank fbank_;

  BlockingQueue<std::vector<float>> feature_queue_;
  int num_frames_;
  bool input_finished_;
  std::vector<float> remained_wav_;

  mutable std::mutex mutex_;
  std::condition_variable finish_condition_;
};

}  // namespace wenet

#endif  // FRONTEND_FEATURE_PIPELINE_H_
