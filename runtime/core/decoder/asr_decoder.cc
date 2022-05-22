// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#include "decoder/asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "utils/timer.h"

namespace wenet {

AsrDecoder::AsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
                       std::shared_ptr<DecodeResource> resource,
                       const DecodeOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      // Make a copy of the model ASR model since we will change the inner
      // status of the model
      model_(resource->model->Copy()),
      post_processor_(resource->post_processor),
      symbol_table_(resource->symbol_table),
      fst_(resource->fst),
      unit_table_(resource->unit_table),
      opts_(opts),
      ctc_endpointer_(new CtcEndpoint(opts.ctc_endpoint_config)) {
  if (opts_.reverse_weight > 0) {
    // Check if model has a right to left decoder
    CHECK(model_->is_bidirectional_decoder());
  }
  if (nullptr == fst_) {
    searcher_.reset(new CtcPrefixBeamSearch(opts.ctc_prefix_search_opts,
                                            resource->context_graph));
  } else {
    searcher_.reset(new CtcWfstBeamSearch(*fst_, opts.ctc_wfst_search_opts,
                                          resource->context_graph));
  }
  ctc_endpointer_->frame_shift_in_ms(frame_shift_in_ms());
}

void AsrDecoder::Reset() {
  start_ = false;
  result_.clear();
  num_frames_ = 0;
  global_frame_offset_ = 0;
  model_->Reset();
  searcher_->Reset();
  feature_pipeline_->Reset();
  ctc_endpointer_->Reset();
}

void AsrDecoder::ResetContinuousDecoding() {
  global_frame_offset_ = num_frames_;
  start_ = false;
  result_.clear();
  model_->Reset();
  searcher_->Reset();
  ctc_endpointer_->Reset();
}

DecodeState AsrDecoder::Decode(bool block) {
  return this->AdvanceDecoding(block);
}

void AsrDecoder::Rescoring() {
  // Do attention rescoring
  Timer timer;
  AttentionRescoring();
  VLOG(2) << "Rescoring cost latency: " << timer.Elapsed() << "ms.";
}

DecodeState AsrDecoder::AdvanceDecoding(bool block) {
  DecodeState state = DecodeState::kEndBatch;
  model_->set_chunk_size(opts_.chunk_size);
  model_->set_num_left_chunks(opts_.num_left_chunks);
  int num_requried_frames = model_->num_frames_for_chunk(start_);
  std::vector<std::vector<float>> chunk_feats;
  // Return immediately if we do not want to block
  if (!block && !feature_pipeline_->input_finished() &&
      feature_pipeline_->NumQueuedFrames() < num_requried_frames) {
    return DecodeState::kWaitFeats;
  }
  // If not okay, that means we reach the end of the input
  if (!feature_pipeline_->Read(num_requried_frames, &chunk_feats)) {
    state = DecodeState::kEndFeats;
  }

  num_frames_ += chunk_feats.size();
  VLOG(2) << "Required " << num_requried_frames << " get "
          << chunk_feats.size();
  Timer timer;
  std::vector<std::vector<float>> ctc_log_probs;
  model_->ForwardEncoder(chunk_feats, &ctc_log_probs);
  int forward_time = timer.Elapsed();
  timer.Reset();
  searcher_->Search(ctc_log_probs);
  int search_time = timer.Elapsed();
  VLOG(3) << "forward takes " << forward_time << " ms, search takes "
          << search_time << " ms";
  UpdateResult();

  if (state != DecodeState::kEndFeats) {
    if (ctc_endpointer_->IsEndpoint(ctc_log_probs, DecodedSomething())) {
      VLOG(1) << "Endpoint is detected at " << num_frames_;
      state = DecodeState::kEndpoint;
    }
  }

  start_ = true;
  return state;
}

void AsrDecoder::UpdateResult(bool finish) {
  const auto& hypotheses = searcher_->Outputs();
  const auto& inputs = searcher_->Inputs();
  const auto& likelihood = searcher_->Likelihood();
  const auto& times = searcher_->Times();
  result_.clear();

  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult path;
    path.score = likelihood[i];
    int offset = global_frame_offset_ * feature_frame_shift_in_ms();
    for (size_t j = 0; j < hypothesis.size(); j++) {
      std::string word = symbol_table_->Find(hypothesis[j]);
      // A detailed explanation of this if-else branch can be found in
      // https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
      if (searcher_->Type() == kWfstBeamSearch) {
        path.sentence += (' ' + word);
      } else {
        path.sentence += (word);
      }
    }

    // TimeStamp is only supported in final result
    // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
    // various FST operations when building the decoding graph. So here we use
    // time stamp of the input(e2e model unit), which is more accurate, and it
    // requires the symbol table of the e2e model used in training.
    if (unit_table_ != nullptr && finish) {
      const std::vector<int>& input = inputs[i];
      const std::vector<int>& time_stamp = times[i];
      CHECK_EQ(input.size(), time_stamp.size());
      for (size_t j = 0; j < input.size(); j++) {
        std::string word = unit_table_->Find(input[j]);
        int start = time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_ > 0
                        ? time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_
                        : 0;
        if (j > 0) {
          start = (time_stamp[j] - time_stamp[j - 1]) * frame_shift_in_ms() <
                          time_stamp_gap_
                      ? (time_stamp[j - 1] + time_stamp[j]) / 2 *
                            frame_shift_in_ms()
                      : start;
        }
        int end = time_stamp[j] * frame_shift_in_ms();
        if (j < input.size() - 1) {
          end = (time_stamp[j + 1] - time_stamp[j]) * frame_shift_in_ms() <
                        time_stamp_gap_
                    ? (time_stamp[j + 1] + time_stamp[j]) / 2 *
                          frame_shift_in_ms()
                    : end;
        }
        WordPiece word_piece(word, offset + start, offset + end);
        path.word_pieces.emplace_back(word_piece);
      }
    }

    if (post_processor_ != nullptr) {
      path.sentence = post_processor_->Process(path.sentence, finish);
    }
    result_.emplace_back(path);
  }

  if (DecodedSomething()) {
    VLOG(1) << "Partial CTC result " << result_[0].sentence;
  }
}

void AsrDecoder::AttentionRescoring() {
  searcher_->FinalizeSearch();
  UpdateResult(true);
  // No need to do rescoring
  if (0.0 == opts_.rescoring_weight) {
    return;
  }
  // Inputs() returns N-best input ids, which is the basic unit for rescoring
  // In CtcPrefixBeamSearch, inputs are the same to outputs
  const auto& hypotheses = searcher_->Inputs();
  int num_hyps = hypotheses.size();
  if (num_hyps <= 0) {
    return;
  }

  std::vector<float> rescoring_score;
  model_->AttentionRescoring(hypotheses, opts_.reverse_weight,
                             &rescoring_score);

  // Combine ctc score and rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    result_[i].score = opts_.rescoring_weight * rescoring_score[i] +
                       opts_.ctc_weight * result_[i].score;
  }
  std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
}

}  // namespace wenet
