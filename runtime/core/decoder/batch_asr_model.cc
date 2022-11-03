// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)
// Copyright (c) 2022 SoundDataConverge Co.LTD (Weiliang Chong)

#include "decoder/batch_asr_model.h"

#include <memory>
#include <utility>

namespace wenet {

void BatchAsrModel::ForwardEncoder(
    const batch_feature_t& batch_feats,
    const std::vector<int>& batch_feats_lens,
    std::vector<std::vector<std::vector<float>>>& batch_topk_scores,
    std::vector<std::vector<std::vector<int32_t>>>& batch_topk_indexs) {
  this->ForwardEncoderFunc(
      batch_feats,
      batch_feats_lens,
      batch_topk_scores,
      batch_topk_indexs);
  }

}  // namespace wenet
