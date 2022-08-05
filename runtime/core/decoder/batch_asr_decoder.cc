// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "decoder/batch_asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "utils/timer.h"

namespace wenet {

BatchAsrDecoder::BatchAsrDecoder(std::shared_ptr<FeaturePipelineConfig> config,
                       std::shared_ptr<DecodeResource> resource,
                       const DecodeOptions& opts)
    : feature_config_(config),
      fbank_(config->num_bins, config->sample_rate, config->frame_length, config->frame_shift),
      model_(resource->batch_model->Copy()),
      post_processor_(resource->post_processor),
      symbol_table_(resource->symbol_table),
      fst_(resource->fst),
      unit_table_(resource->unit_table),
      opts_(opts) {
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
}

void BatchAsrDecoder::Reset() {
  result_.clear();
  batch_result_.clear();
  global_frame_offset_ = 0;
  searcher_->Reset();
}

void BatchAsrDecoder::Decode(const std::vector<std::vector<float>>& wavs) {
  // 1. calc fbank feature of the batch of wavs
  Timer timer;
  batch_feature_t batch_feats;
  std::vector<int> batch_feats_lens;
  VLOG(1) << "wavs : " << wavs.size();
  for (const auto& wav : wavs) {
    VLOG(1) << "wav : " << wav.size();
    feature_t feats;
    int num_frames = fbank_.Compute(wav, &feats);
    VLOG(1) << "feat leng is " << num_frames;
    batch_feats.push_back(std::move(feats));
    batch_feats_lens.push_back(num_frames);
  }
  int feat_time = timer.Elapsed();
  VLOG(1) << "feat_time : " << feat_time;

  // 1.1 feature padding
  timer.Reset();
  int max_len = *std::max_element(batch_feats_lens.begin(), batch_feats_lens.end());
  VLOG(1) << "max length feature : " << max_len;
  for (auto& feat : batch_feats) {
    if (feat.size() == max_len) continue;
    int pad_len = max_len - feat.size();
    for (size_t i = 0; i< pad_len; i++) {
      std::vector<float> one(feature_config_->num_bins, 0);
      feat.push_back(std::move(one));
    }
  }
  VLOG(1) << "padding time : " << timer.Elapsed();
  timer.Reset();

  // 2. encoder forward
  batch_ctc_log_prob_t batch_ctc_log_probs;
  model_->ForwardEncoder(batch_feats, batch_feats_lens, batch_ctc_log_probs);
  VLOG(1) << "encoder forward time : " << timer.Elapsed();

  // 3. ctc search one by one of the batch
  // it seems, decoder forward only support 1 encoder_out with n-best ctc search result
  int batch_size = wavs.size();
  batch_result_.clear();
  for (size_t i = 0; i < batch_size; i++) {
    timer.Reset();
    const auto& ctc_log_probs = batch_ctc_log_probs[i];
    // 3.1. ctc search
    searcher_->Search(ctc_log_probs);
    int search_time = timer.Elapsed();
    VLOG(1) << "search takes " << search_time << " ms";

    // 3.2. rescoring
    timer.Reset();
    AttentionRescoring(i);
    VLOG(1) << "Rescoring cost latency: " << timer.Elapsed() << "ms.";

    // 3.3. save to batch_result_
    batch_result_.push_back(std::move(result_));

    // 3.4 reset
    searcher_->Reset();
  }
}

void BatchAsrDecoder::UpdateResult(bool finish) {
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
}

void BatchAsrDecoder::AttentionRescoring(int batch_index) {
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
  model_->AttentionRescoring(hypotheses, batch_index, opts_.reverse_weight,
                             &rescoring_score);

  // Combine ctc score and rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    result_[i].score = opts_.rescoring_weight * rescoring_score[i] +
                       opts_.ctc_weight * result_[i].score;
  }
  std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
}

}  // namespace wenet
