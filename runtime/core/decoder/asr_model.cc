// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)

#include "decoder/asr_model.h"

#include <memory>
#include <utility>

namespace wenet {

int AsrModel::num_frames_for_chunk(bool start) const {
  int num_requried_frames = 0;
  if (chunk_size_ > 0) {
    if (!start) {                        // First batch
      int context = right_context_ + 1;  // Add current frame
      num_requried_frames = (chunk_size_ - 1) * subsampling_rate_ + context;
    } else {
      num_requried_frames = chunk_size_ * subsampling_rate_;
    }
  } else {
    num_requried_frames = std::numeric_limits<int>::max();
  }
  return num_requried_frames;
}

void AsrModel::CacheFeature(
    const std::vector<std::vector<float>>& chunk_feats) {
  // Cache feature for next chunk
  const int cached_feature_size = 1 + right_context_ - subsampling_rate_;
  if (chunk_feats.size() >= cached_feature_size) {
    // TODO(Binbin Zhang): Only deal the case when
    // chunk_feats.size() > cached_feature_size here, and it's consistent
    // with our current model, refine it later if we have new model or
    // new requirements
    cached_feature_.resize(cached_feature_size);
    for (int i = 0; i < cached_feature_size; ++i) {
      cached_feature_[i] =
          chunk_feats[chunk_feats.size() - cached_feature_size + i];
    }
  }
}

void AsrModel::ForwardEncoder(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* ctc_prob) {
  ctc_prob->clear();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  if (num_frames > right_context_ + 1) {
    this->ForwardEncoderFunc(chunk_feats, ctc_prob);
    this->CacheFeature(chunk_feats);
  }
}

}  // namespace wenet
