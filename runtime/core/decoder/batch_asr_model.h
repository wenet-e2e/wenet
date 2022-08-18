// Copyright 2022 Horizon Robotics. All Rights Reserved.
// Author: binbin.zhang@horizon.ai (Binbin Zhang)

#ifndef DECODER_BATCH_ASR_MODEL_H_
#define DECODER_BATCH_ASR_MODEL_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "utils/timer.h"
#include "utils/utils.h"

namespace wenet {

using feature_t = std::vector<std::vector<float>>;
using batch_feature_t = std::vector<feature_t>;
using ctc_log_prob_t = std::vector<std::vector<float>>;
using batch_ctc_log_prob_t = std::vector<ctc_log_prob_t>;

class BatchAsrModel {

 public:
  virtual int right_context() const { return right_context_; }
  virtual int subsampling_rate() const { return subsampling_rate_; }
  virtual int sos() const { return sos_; }
  virtual int eos() const { return eos_; }
  virtual bool is_bidirectional_decoder() const {
    return is_bidirectional_decoder_;
  }

  virtual void ForwardEncoder(
      const batch_feature_t& batch_feats,
      const std::vector<int>& batch_feats_lens,
      batch_ctc_log_prob_t& batch_ctc_prob);

  virtual void AttentionRescoring(const std::vector<std::vector<std::vector<int>>>& batch_hyps,
                          float reverse_weight,
                          std::vector<std::vector<float>>* attention_scores) = 0;

  virtual std::shared_ptr<BatchAsrModel> Copy() const = 0;

 protected:
  virtual void ForwardEncoderFunc(
      const batch_feature_t& batch_feats,
      const std::vector<int>& batch_feats_lens,
      batch_ctc_log_prob_t& batch_ctc_prob) = 0;

  int right_context_ = 1;
  int subsampling_rate_ = 1;
  int sos_ = 0;
  int eos_ = 0;
  bool is_bidirectional_decoder_ = false;
};

}  // namespace wenet

#endif  // DECODER_BATCH_ASR_MODEL_H_
