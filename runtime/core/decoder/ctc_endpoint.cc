// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#include "decoder/ctc_endpoint.h"

#include <string>

#include "utils/log.h"

namespace wenet {

CtcEndpoint::CtcEndpoint(const CtcEndpointConfig& config) : config_(config) {
  Reset();
}

void CtcEndpoint::Reset() {
  num_frames_decoded_ = 0;
  num_frames_trailing_blank_ = 0;
}

static bool RuleActivated(const CtcEndpointRule& rule,
                          const std::string& rule_name, bool decoded_sth,
                          int trailing_silence, int utterance_length) {
  bool ans = (decoded_sth || !rule.must_decoded_sth) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
  if (ans) {
    VLOG(2) << "Endpointing rule " << rule_name
            << " activated: " << (decoded_sth ? "true" : "false") << ','
            << trailing_silence << ',' << utterance_length;
  }
  return ans;
}

bool CtcEndpoint::IsEndpoint(const torch::Tensor& ctc_log_probs,
                             bool decoded_something) {
  for (int t = 0; t < ctc_log_probs.size(0); ++t) {
    torch::Tensor logp_t = ctc_log_probs[t];
    float blank_prob = expf(logp_t[config_.blank].item<float>());

    num_frames_decoded_++;
    if (blank_prob > config_.blank_threshold) {
      num_frames_trailing_blank_++;
    } else {
      num_frames_trailing_blank_ = 0;
    }
  }
  CHECK_GE(num_frames_decoded_, num_frames_trailing_blank_);
  CHECK_GT(frame_shift_in_ms_, 0);
  int utterance_length = num_frames_decoded_ * frame_shift_in_ms_;
  int trailing_silence = num_frames_trailing_blank_ * frame_shift_in_ms_;
  if (RuleActivated(config_.rule1, "rule1", decoded_something, trailing_silence,
                    utterance_length))
    return true;
  if (RuleActivated(config_.rule2, "rule2", decoded_something, trailing_silence,
                    utterance_length))
    return true;
  if (RuleActivated(config_.rule3, "rule3", decoded_something, trailing_silence,
                    utterance_length))
    return true;
  return false;
}

}  // namespace wenet
