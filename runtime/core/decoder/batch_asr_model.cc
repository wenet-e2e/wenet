// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)

#include "decoder/batch_asr_model.h"

#include <memory>
#include <utility>

namespace wenet {

void BatchAsrModel::ForwardEncoder(
    const batch_feature_t& batch_feats,
    const std::vector<int>& batch_feats_lens,
    batch_ctc_log_prob_t& batch_ctc_prob) {
  batch_ctc_prob.clear();
  this->ForwardEncoderFunc(
      batch_feats,
      batch_feats_lens,
      batch_ctc_prob);
  }

}  // namespace wenet
