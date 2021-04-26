// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_CTC_ENDPOINT_H_
#define DECODER_CTC_ENDPOINT_H_

#include "torch/torch.h"

namespace wenet {

struct CtcEndpointRule {
  bool must_decoded_sth;
  int min_trailing_silence;
  int min_utterance_length;

  CtcEndpointRule(bool must_decoded_sth = true, int min_trailing_silence = 1000,
                  int min_utterance_length = 0)
      : must_decoded_sth(must_decoded_sth),
        min_trailing_silence(min_trailing_silence),
        min_utterance_length(min_utterance_length) {}
};

struct CtcEndpointConfig {
  /// We consider blank as silence for purposes of endpointing.
  int blank = 0;                // blank id
  float blank_threshold = 0.8;  // blank threshold to be silence
  /// We support three rules. We terminate decoding if ANY of these rules
  /// evaluates to "true". If you want to add more rules, do it by changing this
  /// code. If you want to disable a rule, you can set the silence-timeout for
  /// that rule to a very large number.

  /// rule1 times out after 5000 ms of silence, even if we decoded nothing.
  CtcEndpointRule rule1;
  /// rule2 times out after 1000 ms of silence after decoding something.
  CtcEndpointRule rule2;
  /// rule3 times out after the utterance is 20000 ms long, regardless of
  /// anything else.
  CtcEndpointRule rule3;

  CtcEndpointConfig()
      : rule1(false, 5000, 0), rule2(true, 1000, 0), rule3(false, 0, 20000) {}
};

class CtcEndpoint {
 public:
  explicit CtcEndpoint(const CtcEndpointConfig& config);

  void Reset();
  /// This function returns true if this set of endpointing rules thinks we
  /// should terminate decoding.
  bool IsEndpoint(const torch::Tensor& ctc_log_probs, bool decoded_something);

  void frame_shift_in_ms(int frame_shift_in_ms) {
    frame_shift_in_ms_ = frame_shift_in_ms;
  }

 private:
  CtcEndpointConfig config_;
  int frame_shift_in_ms_ = -1;
  int num_frames_decoded_ = 0;
  int num_frames_trailing_blank_ = 0;
};

}  // namespace wenet

#endif  // DECODER_CTC_ENDPOINT_H_
