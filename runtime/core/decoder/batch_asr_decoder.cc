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
#include <thread>
#include <limits>
#include <utility>

#include "utils/timer.h"

namespace wenet {

BatchAsrDecoder::BatchAsrDecoder(std::shared_ptr<FeaturePipelineConfig> config,
                       std::shared_ptr<DecodeResource> resource,
                       const DecodeOptions& opts)
    : feature_config_(config),
      beam_size_(opts.ctc_prefix_search_opts.first_beam_size),
      fbank_(config->num_bins, config->sample_rate, config->frame_length, config->frame_shift),
      model_(resource->batch_model->Copy()),
      post_processor_(resource->post_processor),
      symbol_table_(resource->symbol_table),
      fst_(resource->fst),
      unit_table_(resource->unit_table),
      resource_(resource),
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
  batch_result_.clear();
  searcher_->Reset();
}

void BatchAsrDecoder::SearchWorker(const ctc_log_prob_t& ctc_log_probs, int index) {
  Timer ctc_timer;
  std::unique_ptr<SearchInterface> searcher;
  if (nullptr == fst_) {
    searcher.reset(new CtcPrefixBeamSearch(opts_.ctc_prefix_search_opts,
                                            resource_->context_graph));
  } else {
    searcher.reset(new CtcWfstBeamSearch(*fst_, opts_.ctc_wfst_search_opts,
                                          resource_->context_graph));
  }
  // 3.1. ctc search
  ctc_timer.Reset();
  searcher->Search(ctc_log_probs);
  searcher->FinalizeSearch();
  std::vector<DecodeResult> result;
  UpdateResult(searcher.get(), result);
  VLOG(1) << "\tctc search i==" << index << " takes " << ctc_timer.Elapsed() << " ms";
  std::lock_guard<std::mutex> lock(mutex_);
  batch_pair_result_.emplace_back(std::make_pair(index, std::move(result)));
  const auto& hypotheses = searcher->Inputs();
  if (hypotheses.size() < beam_size_) {
    VLOG(2) << "=== searcher->Inputs() size < beam_size_, padding...";
    std::vector<std::vector<int>> hyps = hypotheses;
    int to_pad = beam_size_ - hypotheses.size();
    for (size_t i = 0; i < to_pad; i++) {
      std::vector<int> pad = {0};
      hyps.push_back(std::move(pad));
    }
    batch_hyps_.emplace_back(std::make_pair(index, std::move(hyps)));
  } else {
    batch_hyps_.emplace_back(std::make_pair(index, std::move(hypotheses)));
  }
}

void BatchAsrDecoder::FbankWorker(const std::vector<float>& wav, int index) {
  Timer timer;
  feature_t feats;
  int num_frames = fbank_.Compute(wav, &feats);
  std::lock_guard<std::mutex> lock(mutex_);
  batch_feats_.push_back(std::make_pair(index, std::move(feats)));
  batch_feats_lens_.push_back(std::make_pair(index, num_frames));
  VLOG(1) << "\tfeature comput i==" << index << ", takes " << timer.Elapsed() << " ms.";
}

void BatchAsrDecoder::Decode(const std::vector<std::vector<float>>& wavs) {
  // 1. calc fbank feature of the batch of wavs
  Timer timer;
  batch_feature_t batch_feats;
  std::vector<int> batch_feats_lens;
  if (wavs.size() > 1) {
    std::vector<std::thread> fbank_threads;
    for (size_t i = 0; i < wavs.size(); i++) {
      const std::vector<float>& wav = wavs[i];
      std::thread thd(&BatchAsrDecoder::FbankWorker, this, wav, i);
      fbank_threads.push_back(std::move(thd));
    }
    for(auto& thd : fbank_threads) {
      thd.join();
    }
    std::sort(batch_feats_.begin(), batch_feats_.end());
    std::sort(batch_feats_lens_.begin(), batch_feats_lens_.end());
    for (auto& pair : batch_feats_) {
      batch_feats.push_back(std::move(pair.second));
    }
    for (auto& pair : batch_feats_lens_) {
      batch_feats_lens.push_back(pair.second);
    }
  } else {
    // only one wave
    feature_t feats;
    int num_frames = fbank_.Compute(wavs[0], &feats);
    batch_feats.push_back(feats);
    batch_feats_lens.push_back(num_frames);
  }
  VLOG(1) << "feature Compute takes " << timer.Elapsed() << " ms.";

  // 1.1 feature padding
  if (wavs.size() > 1) {
    timer.Reset();
    int max_len = *std::max_element(batch_feats_lens.begin(), batch_feats_lens.end());
    for (auto& feat : batch_feats) {
      if (feat.size() == max_len) continue;
      int pad_len = max_len - feat.size();
      for (size_t i = 0; i< pad_len; i++) {
        std::vector<float> one(feature_config_->num_bins, 0.0);
        feat.push_back(std::move(one));
      }
    }
    VLOG(1) << "padding feautre takes " << timer.Elapsed() << " ms.";
  }

  // 2. encoder forward
  timer.Reset();
  batch_ctc_log_prob_t batch_ctc_log_probs;
  model_->ForwardEncoder(batch_feats, batch_feats_lens, batch_ctc_log_probs);
  VLOG(1) << "encoder forward takes " << timer.Elapsed() << " ms.";

  // 3. ctc search one by one of the batch
  // create batch of tct search result for attention decoding
  timer.Reset();
  int batch_size = wavs.size();
  std::vector<std::vector<std::vector<int>>> batch_hyps;
  if (batch_size > 1) {
    batch_pair_result_.clear();
    batch_hyps_.clear();
    std::vector<std::thread> search_threads;
    for (size_t i = 0; i < batch_size; i++) {
      const auto& ctc_log_probs = batch_ctc_log_probs[i];
      std::thread thd(&BatchAsrDecoder::SearchWorker, this, ctc_log_probs, i);
      search_threads.push_back(std::move(thd));
    }
    for(auto& thd : search_threads) {
      thd.join();
    }
    std::sort(batch_hyps_.begin(), batch_hyps_.end());
    std::sort(batch_pair_result_.begin(), batch_pair_result_.end(), [](auto& a, auto& b) {
        return a.first < b.first; });
    for (auto& pair : batch_hyps_) {
      batch_hyps.push_back(std::move(pair.second));
    }
    batch_result_.clear();
    for (auto& pair : batch_pair_result_) {
      batch_result_.push_back(std::move(pair.second));
    }
  } else {
    // one wav
    VLOG(1) << "=== ctc search for one wav! " << batch_ctc_log_probs[0].size();
    searcher_->Search(batch_ctc_log_probs[0]);
    searcher_->FinalizeSearch();
    std::vector<DecodeResult> result;
    UpdateResult(searcher_.get(), result);
    batch_result_.push_back(std::move(result));
    const auto& hypotheses = searcher_->Inputs();
    if (hypotheses.size() < beam_size_) {
      VLOG(2) << "=== searcher->Inputs() size < beam_size_, padding...";
      std::vector<std::vector<int>> hyps = hypotheses;
      int to_pad = beam_size_ - hypotheses.size();
      for (size_t i = 0; i < to_pad; i++) {
        std::vector<int> pad = {0};
        hyps.push_back(std::move(pad));
      }
      batch_hyps.push_back(std::move(hyps));
    } else {
      batch_hyps.push_back(std::move(hypotheses));
    }
  }
  VLOG(1) << "ctc search batch(" << batch_size << ") takes " << timer.Elapsed() << " ms.";
  std::vector<std::vector<float>> ctc_scores(batch_size);
  for (int i = 0; i < batch_result_.size(); ++i) {
    ctc_scores[i].resize(beam_size_);
    for (int j = 0; j < beam_size_; ++j) {
      ctc_scores[i][j] = batch_result_[i][j].score;
    }
  }
  // 4. attention rescoring
  timer.Reset();
  std::vector<std::vector<float>> attention_scores;
  model_->AttentionRescoring(batch_hyps, ctc_scores, attention_scores);
  VLOG(1) << "attention rescoring takes " << timer.Elapsed() << " ms.";
  for (size_t i = 0; i < batch_size; i++) {
    std::vector<DecodeResult>& result = batch_result_[i];
    for (size_t j = 0; j < beam_size_; j++) {
      result[j].score = attention_scores[i][j];
    }
    std::sort(result.begin(), result.end(), DecodeResult::CompareFunc);
  }
}

void BatchAsrDecoder::UpdateResult(SearchInterface* searcher, std::vector<DecodeResult>& result) {
  bool finish = true;
  const auto& hypotheses = searcher->Outputs();
  const auto& inputs = searcher->Inputs();
  const auto& likelihood = searcher->Likelihood();
  const auto& times = searcher->Times();
  result.clear();

  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult path;
    path.score = likelihood[i];
    for (size_t j = 0; j < hypothesis.size(); j++) {
      std::string word = symbol_table_->Find(hypothesis[j]);
      // A detailed explanation of this if-else branch can be found in
      // https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
      if (searcher->Type() == kWfstBeamSearch) {
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
        WordPiece word_piece(word, start, end);
        path.word_pieces.emplace_back(word_piece);
      }
    }

    if (post_processor_ != nullptr) {
      path.sentence = post_processor_->Process(path.sentence, finish);
    }
    result.emplace_back(path);
  }
}

const std::string BatchAsrDecoder::get_batch_result(int nbest, bool enable_timestamp) {
    json::JSON obj;
    obj["status"] = "ok";
    obj["type"] = "final_result";
    obj["batch_size"] = batch_result_.size();
    obj["batch_result"] = json::Array();
    for (const auto& result : batch_result_) {
      json::JSON batch_one;
      batch_one["nbest"] = json::Array();
      for (int i = 0; i < nbest && i < result.size(); i++) {
        json::JSON one;
        one["sentence"] = result[i].sentence;
        // one["score"] = result[i].score;
        if (enable_timestamp) {
          one["word_pieces"] = json::Array();
          for (const auto& word_piece : result[i].word_pieces) {
            json::JSON piece;
            piece["word"] = word_piece.word;
            piece["start"] = word_piece.start;
            piece["end"] = word_piece.end;
            one["word_pieces"].append(piece);
          }
        }
        one["sentence"] = result[i].sentence;
        batch_one["nbest"].append(one);
      }
      obj["batch_result"].append(batch_one);
    }
    return obj.dump();
  }

}  // namespace wenet
