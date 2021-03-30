// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#include "decoder/online_endpoint.h"

#include <string>

#include "glog/logging.h"

namespace wenet {

static bool RuleActivated(const OnlineEndpointRule &rule,
                          const std::string &rule_name,
                          bool contains_nonsilence,
                          int trailing_silence,
                          int utterance_length) {
  bool ans = (contains_nonsilence || !rule.must_contain_nonsilence) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
  if (ans) {
    VLOG(2) << "Endpointing rule " << rule_name << " activated: "
                  << (contains_nonsilence ? "true" : "false") << ','
                  << trailing_silence << ',' << utterance_length;
  }
  return ans;
}

bool EndpointDetected(const OnlineEndpointConfig& config,
                      bool contains_nonsilence,
                      int num_frames_decoded,
                      int trailing_silence_frames,
                      int frame_shift_in_ms) {
  CHECK_GE(num_frames_decoded, trailing_silence_frames);

  int utterance_length = num_frames_decoded * frame_shift_in_ms;
  int trailing_silence = trailing_silence_frames * frame_shift_in_ms;

  if (RuleActivated(config.rule1, "rule1", contains_nonsilence,
                    trailing_silence, utterance_length))
    return true;
  if (RuleActivated(config.rule2, "rule2", contains_nonsilence,
                    trailing_silence, utterance_length))
    return true;
  if (RuleActivated(config.rule3, "rule3", contains_nonsilence,
                    trailing_silence, utterance_length))
    return true;
  return false;
}

}  // namespace wenet
