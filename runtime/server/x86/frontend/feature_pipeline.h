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

#include <glog/logging.h>

#include <string>
#include <vector>
#include <queue>

#include "frontend/fbank.h"

#ifndef WENET_FRONTEND_FEATURE_PIPELINE_H_
#define WENET_FRONTEND_FEATURE_PIPELINE_H_

namespace wenet {

struct FeaturePipelineConfig {
  int num_bins;
  int sample_rate;
  int frame_length;
  int frame_shift;
  FeaturePipelineConfig():
      num_bins(40),  // 40 dim fbank
      sample_rate(16000),  // 16k sample rate
      frame_length(400),  // frame length 25ms,
      frame_shift(160) {
  }

  void Info() const {
    LOG(INFO) << "feature pipeline config"
              << " num_bins " << num_bins
              << " frame_length " << frame_length
              << "frame_shift" << frame_shift;
  }
};

class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);

  void AcceptWaveform(const std::vector<float>& wav);
  int NumFramesReady() const { return num_frames_; }
  void InputFinished() {
    CHECK(!input_finished_);
    input_finished_ = true;
  }
  int FeatureDim() const {
    return feature_dim_;
  }
  int Read(int num_frames, std::vector<std::vector<float> >* feats);
  void Reset();
  bool IsLastFrame(int frame) const {
    return input_finished_ && (frame == num_frames_ - 1);
  }

 private:
  const FeaturePipelineConfig &config_;
  int feature_dim_;
  Fbank fbank_;
  // Feature queue
  std::queue<std::vector<float> > feature_queue_;
  int num_frames_;
  bool input_finished_;
  std::vector<float> remained_wav_;
};

}  // namespace wenet

#endif  // WENET_FRONTEND_FEATURE_PIPELINE_H_


