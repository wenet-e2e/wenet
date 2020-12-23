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

#include <algorithm>

#include "frontend/feature_pipeline.h"

namespace wenet {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config):
    config_(config),
    left_context_(config.left_context),
    right_context_(config.right_context),
    raw_feat_dim_(config.num_bins),
    fbank_(config.num_bins, config.sample_rate,
           config.frame_length, config.frame_shift),
    num_frames_(0),
    done_(false) {
}

void FeaturePipeline::AcceptRawWav(const std::vector<float>& wav) {
  std::vector<float> feat;
  std::vector<float> waves;
  waves.insert(waves.end(), ctx_wav_.begin(), ctx_wav_.end());
  waves.insert(waves.end(), wav.begin(), wav.end());
  int num_frames = fbank_.Compute(waves, &feat);
  if (feature_buf_.size() == 0 && left_context_ > 0) {
    for (int i = 0; i < left_context_; i++) {
      feature_buf_.insert(feature_buf_.end(), feat.begin(),
                          feat.begin() + raw_feat_dim_);
    }
  }
  feature_buf_.insert(feature_buf_.end(), feat.begin(), feat.end());
  num_frames_ += num_frames;

  int left_samples = waves.size() - config_.frame_shift * num_frames;
  ctx_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames,
            waves.end(), ctx_wav_.begin());
}

int FeaturePipeline::NumFramesReady() const {
  if (num_frames_ < right_context_) return 0;
  if (done_) {
    return num_frames_;
  } else {
    return num_frames_ - right_context_;
  }
}

void FeaturePipeline::SetDone() {
  CHECK(!done_);
  done_ = true;
  if (num_frames_ == 0) return;
  // copy last frames to buffer
  std::vector<float> last_feat(feature_buf_.end() - raw_feat_dim_,
                               feature_buf_.end());
  for (int i = 0; i < right_context_; i++) {
    feature_buf_.insert(feature_buf_.end(), last_feat.begin(), last_feat.end());
  }
}

int FeaturePipeline::ReadFeature(int t, std::vector<float>* feat) {
  CHECK(t < num_frames_);
  int num_frames_ready = NumFramesReady();
  if (num_frames_ready <= 0) return 0;
  int total_frame = num_frames_ready - t;
  int feat_dim = (left_context_ + 1 + right_context_) * raw_feat_dim_;
  feat->resize(total_frame * feat_dim);
  for (int i = t; i < num_frames_ready; i++) {
    memcpy(feat->data() + (i - t) * feat_dim,
           feature_buf_.data() + i * raw_feat_dim_,
           sizeof(float) * feat_dim);
  }
  return total_frame;
}

int FeaturePipeline::ReadOneFrame(int t, float *data) {
  CHECK(data != nullptr);
  CHECK(t < num_frames_);
  int num_frames_ready = NumFramesReady();
  if (num_frames_ready <= 0) return 0;
  CHECK(t <= num_frames_ready);
  int feat_dim = (left_context_ + 1 + right_context_) * raw_feat_dim_;
  memcpy(data, feature_buf_.data() + t * raw_feat_dim_,
         sizeof(float) * feat_dim);
  return 1;
}

int FeaturePipeline::ReadAllFeature(std::vector<float> *feat) {
  return ReadFeature(0, feat);
}

int FeaturePipeline::NumFrames(int size) const {
  return 1 + (size - config_.frame_length) / config_.frame_shift;
}

}  // namespace wenet

