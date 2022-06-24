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

#include "frontend/feature_pipeline.h"

#include <algorithm>
#include <utility>

namespace wenet {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config)
    : config_(config),
      feature_dim_(config.num_bins),
      fbank_(config.num_bins, config.sample_rate, config.frame_length,
             config.frame_shift),
      num_frames_(0),
      wait_frames_(0),
      input_finished_(false) {}

void FeaturePipeline::AcceptWaveform(const float* pcm, const int size) {
  std::vector<std::vector<float>> feats;
  std::vector<float> waves;
  waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
  waves.insert(waves.end(), pcm, pcm + size);
  int num_frames = fbank_.Compute(waves, &feats);
  feature_queue_.Push(std::move(feats));
  {
    std::lock_guard<std::mutex> lock(wait_mutex_);
    wait_frames_ += num_frames;
  }
  num_frames_ += num_frames;

  int left_samples = waves.size() - config_.frame_shift * num_frames;
  remained_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames, waves.end(),
            remained_wav_.begin());
  // We are still adding wave, notify input is not finished
  finish_condition_.notify_one();
}

void FeaturePipeline::AcceptWaveform(const int16_t* pcm, const int size) {
  auto* float_pcm = new float[size];
  for (size_t i = 0; i < size; i++) {
    float_pcm[i] = static_cast<float>(pcm[i]);
  }
  this->AcceptWaveform(float_pcm, size);
  delete[] float_pcm;
}

void FeaturePipeline::set_input_finished() {
  CHECK(!input_finished_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_finished_ = true;
  }
  finish_condition_.notify_one();
}

bool FeaturePipeline::ReadFromQueue(std::vector<std::vector<float>>* feats) {
  if (!feature_queue_.Empty()) {
    *feats = std::move(feature_queue_.Pop());
    return true;
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      if (!feature_queue_.Empty()) {
        *feats = std::move(feature_queue_.Pop());
        return true;
      }
    }
    CHECK(input_finished_);
    // Double check queue.empty, see issue#893 for detailed discussions.
    if (!feature_queue_.Empty()) {
      *feats = std::move(feature_queue_.Pop());
      return true;
    } else {
      return false;
    }
  }
}

bool FeaturePipeline::Read(int num_frames,
                           std::vector<std::vector<float>>* feats) {
  bool b = true;
  feats->clear();
  std::vector<std::vector<float>> tmp_feats;
  std::vector<std::vector<float>> chunk_feats;
  tmp_feats.insert(tmp_feats.end(), remained_feats_.begin(),
                   remained_feats_.end());
  while (tmp_feats.size() < num_frames) {
    if (ReadFromQueue(&chunk_feats)) {
      tmp_feats.insert(tmp_feats.end(), chunk_feats.begin(), chunk_feats.end());
    } else {
      b = false;
      break;
    }
  }
  if (num_frames > tmp_feats.size()) {
    num_frames = tmp_feats.size();
  }
  if (num_frames > 0) {
    feats->insert(feats->end(), tmp_feats.begin(),
                  tmp_feats.begin() + num_frames);
    remained_feats_.resize(tmp_feats.size() - num_frames);
    std::copy(tmp_feats.begin() + num_frames, tmp_feats.end(),
              remained_feats_.begin());
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      wait_frames_ -= num_frames;
    }
  }
  return b;
}

int FeaturePipeline::NumQueuedFrames() const {
  std::lock_guard<std::mutex> lock(wait_mutex_);
  return wait_frames_;
}

void FeaturePipeline::Reset() {
  input_finished_ = false;
  num_frames_ = 0;
  remained_wav_.clear();
  remained_feats_.clear();
  feature_queue_.Clear();
  {
    std::lock_guard<std::mutex> lock(wait_mutex_);
    wait_frames_ = 0;
  }
}
bool FeaturePipeline::ReadOne(std::vector<float>* feat) {
  feat->clear();
  std::vector<std::vector<float>> tmp_feats;
  bool b = Read(1, &tmp_feats);
  if (!tmp_feats.empty()) {
    *feat = std::move(tmp_feats[0]);
  }
  return b;
}

}  // namespace wenet
