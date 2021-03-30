// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_ONLINE_ENDPOINT_H_
#define DECODER_ONLINE_ENDPOINT_H_

namespace wenet {

struct OnlineEndpointRule {
  bool must_contain_nonsilence;
  int min_trailing_silence;
  int min_utterance_length;

  OnlineEndpointRule(bool must_contain_nonsilence = true,
                     int min_trailing_silence = 1000,
                     int min_utterance_length = 0):
     must_contain_nonsilence(must_contain_nonsilence),
     min_trailing_silence(min_trailing_silence),
     min_utterance_length(min_utterance_length) { }
};

struct OnlineEndpointConfig {
  /// We support three rules. We terminate decoding if ANY of these rules
  /// evaluates to "true". If you want to add more rules, do it by changing this
  /// code. If you want to disable a rule, you can set the silence-timeout for
  /// that rule to a very large number.

  /// rule1 times out after 5000 ms of silence, even if we decoded nothing.
  OnlineEndpointRule rule1;
  /// rule2 times out after 2000 mss of silence after decoding something.
  OnlineEndpointRule rule2;
  /// rule3 times out after the utterance is 20000 ms long, regardless of
  /// anything else.
  OnlineEndpointRule rule3;

  OnlineEndpointConfig():
      rule1(false, 5000, 0),
      rule2(true, 2000, 0),
      rule3(false, 0, 20000) { }
};

/// This function returns true if this set of endpointing rules thinks we
/// should terminate decoding.
bool EndpointDetected(const OnlineEndpointConfig& config,
                      bool contains_nonsilence,
                      int num_frames_decoded,
                      int trailing_silence_frames,
                      int frame_shift_in_ms);
}  // namespace wenet

#endif  // DECODER_ONLINE_ENDPOINT_H_
