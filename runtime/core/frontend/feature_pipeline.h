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

#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "frontend/fbank.h"
#include "utils/blocking_queue.h"
#include "utils/log.h"

namespace wenet {

enum class FeatureType {
  kKaldi = 0,
  kWhisper,
};

struct FeaturePipelineConfig {
  int num_bins;
  int sample_rate;
  int frame_length;
  int frame_shift;
  float low_freq;
  bool pre_emphasis;
  bool scale_input_to_unit;
  float log_floor;
  LogBase log_base;
  WindowType window_type;
  MelType mel_type;
  NormalizationType norm_type;

  FeaturePipelineConfig(int num_bins, int sample_rate,
                        FeatureType feat_type = FeatureType::kKaldi)
      : num_bins(num_bins),                  // 80 dim fbank
        sample_rate(sample_rate) {           // 16k sample rate
    frame_length = sample_rate / 1000 * 25;  // frame length 25ms
    frame_shift = sample_rate / 1000 * 10;   // frame shift 10ms
    if (feat_type == FeatureType::kKaldi) {
      low_freq = 20.0;
      pre_emphasis = true;
      log_floor = std::numeric_limits<float>::epsilon();
      log_base = LogBase::kBaseE;
      window_type = WindowType::kPovey;
      mel_type = MelType::kHTK;
      norm_type = NormalizationType::kKaldi;
      scale_input_to_unit = false;
    } else if (feat_type == FeatureType::kWhisper) {
      low_freq = 0.0;
      pre_emphasis = false;
      log_floor = 1e-10;
      log_base = LogBase::kBase10;
      window_type = WindowType::kHanning;
      mel_type = MelType::kSlaney;
      scale_input_to_unit = true;
      norm_type = NormalizationType::kWhisper;
    }
  }

  void Info() const {
    LOG(INFO) << "feature pipeline config"
              << " num_bins " << num_bins << " frame_length " << frame_length
              << " frame_shift " << frame_shift << " low_freq " << low_freq
              << " preemphasis " << pre_emphasis << " log_floor " << log_floor
              << " log_base " << int(log_base) << " window_type "
              << int(window_type) << " mel_type " << int(mel_type)
              << " norm_type " << int(norm_type);
  }
};

// Typically, FeaturePipeline is used in two threads: one thread A calls
// AcceptWaveform() to add raw wav data and set_input_finished() to notice
// the end of input wav, another thread B (decoder thread) calls Read() to
// consume features.So a BlockingQueue is used to make this class thread safe.

// The Read() is designed as a blocking method when there is no feature
// in feature_queue_ and the input is not finished.

// See bin/decoder_main.cc, websocket/websocket_server.cc and
// decoder/torch_asr_decoder.cc for usage

class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);

  // The feature extraction is done in AcceptWaveform().
  void AcceptWaveform(const float* pcm, const int size);
  void AcceptWaveform(const int16_t* pcm, const int size);

  // Current extracted frames number.
  int num_frames() const { return num_frames_; }
  int feature_dim() const { return feature_dim_; }
  const FeaturePipelineConfig& config() const { return config_; }

  // The caller should call this method when speech input is end.
  // Never call AcceptWaveform() after calling set_input_finished() !
  void set_input_finished();
  bool input_finished() const { return input_finished_; }

  // Return False if input is finished and no feature could be read.
  // Return True if a feature is read.
  // This function is a blocking method. It will block the thread when
  // there is no feature in feature_queue_ and the input is not finished.
  bool ReadOne(std::vector<float>* feat);

  // Read #num_frames frame features.
  // Return False if less than #num_frames features are read and the
  // input is finished.
  // Return True if #num_frames features are read.
  // This function is a blocking method when there is no feature
  // in feature_queue_ and the input is not finished.
  bool Read(int num_frames, std::vector<std::vector<float>>* feats);

  void Reset();
  bool IsLastFrame(int frame) const {
    return input_finished_ && (frame == num_frames_ - 1);
  }

  int NumQueuedFrames() const { return feature_queue_.Size(); }

 private:
  const FeaturePipelineConfig& config_;
  int feature_dim_;
  Fbank fbank_;

  BlockingQueue<std::vector<float>> feature_queue_;
  int num_frames_;
  bool input_finished_;

  // The feature extraction is done in AcceptWaveform().
  // This waveform sample points are consumed by frame size.
  // The residual waveform sample points after framing are
  // kept to be used in next AcceptWaveform() calling.
  std::vector<float> remained_wav_;

  // Used to block the Read when there is no feature in feature_queue_
  // and the input is not finished.
  mutable std::mutex mutex_;
  std::condition_variable finish_condition_;
};

}  // namespace wenet

#endif  // FRONTEND_FEATURE_PIPELINE_H_
