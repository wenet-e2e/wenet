// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#include "decoder/torch_asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "decoder/ctc_endpoint.h"
#include "utils/string.h"
#include "utils/timer.h"

namespace wenet {

TorchAsrDecoder::TorchAsrDecoder(
    std::shared_ptr<FeaturePipeline> feature_pipeline,
    std::shared_ptr<DecodeResource> resource, const DecodeOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      model_(resource->model),
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
    searcher_.reset(new CtcPrefixBeamSearch(opts.ctc_prefix_search_opts));
  } else {
    searcher_.reset(new CtcWfstBeamSearch(*fst_, opts.ctc_wfst_search_opts));
  }
  ctc_endpointer_->frame_shift_in_ms(frame_shift_in_ms());
  InitPostProcessing();
}

void TorchAsrDecoder::InitPostProcessing() {
  fst::SymbolTableIterator iter(*symbol_table_);
  std::string space_symbol = kSpaceSymbol;
  while (!iter.Done()) {
    if (iter.Symbol().size() > space_symbol.size() &&
        std::equal(space_symbol.begin(), space_symbol.end(),
                   iter.Symbol().begin())) {
      wp_start_with_space_symbol_ = true;
      break;
    }
    iter.Next();
  }
}

void TorchAsrDecoder::Reset() {
  start_ = false;
  result_.clear();
  offset_ = 0;
  num_frames_ = 0;
  global_frame_offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  subsampling_cache_ = std::move(torch::jit::IValue());
  elayers_output_cache_ = std::move(torch::jit::IValue());
  conformer_cnn_cache_ = std::move(torch::jit::IValue());
  encoder_outs_.clear();
  cached_feature_.clear();
  searcher_->Reset();
  feature_pipeline_->Reset();
  ctc_endpointer_->Reset();
}

void TorchAsrDecoder::ResetContinuousDecoding() {
  global_frame_offset_ = num_frames_;
  start_ = false;
  result_.clear();
  offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  subsampling_cache_ = std::move(torch::jit::IValue());
  elayers_output_cache_ = std::move(torch::jit::IValue());
  conformer_cnn_cache_ = std::move(torch::jit::IValue());
  encoder_outs_.clear();
  cached_feature_.clear();
  searcher_->Reset();
  ctc_endpointer_->Reset();
}

DecodeState TorchAsrDecoder::Decode() { return this->AdvanceDecoding(); }

void TorchAsrDecoder::Rescoring() {
  // Do attention rescoring
  Timer timer;
  AttentionRescoring();
  LOG(INFO) << "Rescoring cost latency: " << timer.Elapsed() << "ms.";
}

DecodeState TorchAsrDecoder::AdvanceDecoding() {
  DecodeState state = DecodeState::kEndBatch;
  const int subsampling_rate = model_->subsampling_rate();
  const int right_context = model_->right_context();
  const int cached_feature_size = 1 + right_context - subsampling_rate;
  const int feature_dim = feature_pipeline_->feature_dim();
  int num_requried_frames = 0;
  // If opts_.chunk_size > 0, streaming case, read feature chunk by chunk
  // otherwise, none streaming case, read all feature at once
  if (opts_.chunk_size > 0) {
    if (!start_) {                      // First batch
      int context = right_context + 1;  // Add current frame
      num_requried_frames = (opts_.chunk_size - 1) * subsampling_rate + context;
    } else {
      num_requried_frames = opts_.chunk_size * subsampling_rate;
    }
  } else {
    num_requried_frames = std::numeric_limits<int>::max();
  }
  std::vector<std::vector<float>> chunk_feats;
  // If not okay, that means we reach the end of the input
  if (!feature_pipeline_->Read(num_requried_frames, &chunk_feats)) {
    state = DecodeState::kEndFeats;
  }

  num_frames_in_current_chunk_ = chunk_feats.size();
  num_frames_ += chunk_feats.size();
  LOG(INFO) << "Required " << num_requried_frames << " get "
            << chunk_feats.size();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  // The total frames should be big enough to get just one output
  if (num_frames >= right_context + 1) {
    // 1. Prepare libtorch required data, splice cached_feature_ and chunk_feats
    // The first dimension is for batchsize, which is 1.
    torch::Tensor feats =
        torch::zeros({1, num_frames, feature_dim}, torch::kFloat);
    for (size_t i = 0; i < cached_feature_.size(); ++i) {
      torch::Tensor row = torch::from_blob(cached_feature_[i].data(),
                                           {feature_dim}, torch::kFloat)
                              .clone();
      feats[0][i] = std::move(row);
    }
    for (size_t i = 0; i < chunk_feats.size(); ++i) {
      torch::Tensor row =
          torch::from_blob(chunk_feats[i].data(), {feature_dim}, torch::kFloat)
              .clone();
      feats[0][cached_feature_.size() + i] = std::move(row);
    }

    Timer timer;
    // 2. Encoder chunk forward
    int requried_cache_size = opts_.chunk_size * opts_.num_left_chunks;
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {feats,
                                              offset_,
                                              requried_cache_size,
                                              subsampling_cache_,
                                              elayers_output_cache_,
                                              conformer_cnn_cache_};
    // Refer interfaces in wenet/transformer/asr_model.py
    auto outputs = model_->torch_model()
                       ->get_method("forward_encoder_chunk")(inputs)
                       .toTuple()
                       ->elements();
    CHECK_EQ(outputs.size(), 4);
    torch::Tensor chunk_out = outputs[0].toTensor();
    subsampling_cache_ = outputs[1];
    elayers_output_cache_ = outputs[2];
    conformer_cnn_cache_ = outputs[3];
    offset_ += chunk_out.size(1);

    // The first dimension of returned value is for batchsize, which is 1
    torch::Tensor ctc_log_probs = model_->torch_model()
                                      ->run_method("ctc_activation", chunk_out)
                                      .toTensor()[0];
    encoder_outs_.push_back(std::move(chunk_out));
    int forward_time = timer.Elapsed();
    timer.Reset();
    searcher_->Search(ctc_log_probs);
    int search_time = timer.Elapsed();
    VLOG(3) << "forward takes " << forward_time << " ms, search takes "
            << search_time << " ms";
    UpdateResult();

    if (ctc_endpointer_->IsEndpoint(ctc_log_probs, DecodedSomething())) {
      LOG(INFO) << "Endpoint is detected at " << num_frames_;
      state = DecodeState::kEndpoint;
    }

    // 3. Cache feature for next chunk
    if (state == DecodeState::kEndBatch) {
      // TODO(Binbin Zhang): Only deal the case when
      // chunk_feats.size() > cached_feature_size_ here, and it's consistent
      // with our current model, refine it later if we have new model or
      // new requirements
      CHECK(chunk_feats.size() >= cached_feature_size);
      cached_feature_.resize(cached_feature_size);
      for (int i = 0; i < cached_feature_size; ++i) {
        cached_feature_[i] = std::move(
            chunk_feats[chunk_feats.size() - cached_feature_size + i]);
      }
    }
  }

  start_ = true;
  return state;
}

void TorchAsrDecoder::UpdateResult(bool finish) {
  const auto& hypotheses = searcher_->Outputs();
  const auto& inputs = searcher_->Inputs();
  const auto& likelihood = searcher_->Likelihood();
  const auto& times = searcher_->Times();
  result_.clear();

  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult path;
    bool is_englishword_prev = false;
    path.score = likelihood[i];
    int offset = global_frame_offset_ * feature_frame_shift_in_ms();
    for (size_t j = 0; j < hypothesis.size(); j++) {
      std::string word = symbol_table_->Find(hypothesis[j]);
      if (wp_start_with_space_symbol_) {
        path.sentence += word;
        continue;
      }
      bool is_englishword_now = CheckEnglishWord(word);
      if (is_englishword_prev && is_englishword_now) {
        path.sentence += (' ' + word);
      } else {
        path.sentence += (word);
      }
      is_englishword_prev = is_englishword_now;
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
        int start = j > 0 ? time_stamp[j - 1] * frame_shift_in_ms() : 0;
        int end = time_stamp[j] * frame_shift_in_ms();
        WordPiece word_piece(word, offset + start, offset + end);
        path.word_pieces.emplace_back(word_piece);
        start = word_piece.end;
      }
    }
    path.sentence = ProcessBlank(path.sentence);
    result_.emplace_back(path);
  }

  if (DecodedSomething()) {
    VLOG(1) << "Partial CTC result " << result_[0].sentence;
  }
}

float TorchAsrDecoder::AttentionDecoderScore(const torch::Tensor& prob,
                                             const std::vector<int>& hyp,
                                             int eos) {
  float score = 0.0f;
  auto accessor = prob.accessor<float, 2>();
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += accessor[j][hyp[j]];
  }
  score += accessor[hyp.size()][eos];
  return score;
}

void TorchAsrDecoder::AttentionRescoring() {
  searcher_->FinalizeSearch();
  UpdateResult(true);
  // No need to do rescoring
  if (0.0 == opts_.rescoring_weight) {
    return;
  }
  // No encoder output
  if (encoder_outs_.size() == 0) {
    return;
  }

  int sos = model_->sos();
  int eos = model_->eos();
  // Inputs() returns N-best input ids, which is the basic unit for rescoring
  // In CtcPrefixBeamSearch, inputs are the same to outputs
  const auto& hypotheses = searcher_->Inputs();
  int num_hyps = hypotheses.size();
  if (num_hyps <= 0) {
    return;
  }

  torch::NoGradGuard no_grad;
  // Step 1: Prepare input for libtorch
  torch::Tensor hyps_length = torch::zeros({num_hyps}, torch::kLong);
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hypotheses[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_length[i] = static_cast<int64_t>(length);
  }
  torch::Tensor hyps_tensor =
      torch::zeros({num_hyps, max_hyps_len}, torch::kLong);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    hyps_tensor[i][0] = sos;
    for (size_t j = 0; j < hyp.size(); ++j) {
      hyps_tensor[i][j + 1] = hyp[j];
    }
  }

  // Step 2: Forward attention decoder by hyps and corresponding encoder_outs_
  torch::Tensor encoder_out = torch::cat(encoder_outs_, 1);
  auto outputs =
      model_->torch_model()
          ->run_method("forward_attention_decoder", hyps_tensor, hyps_length,
                       encoder_out, opts_.reverse_weight)
          .toTuple()
          ->elements();
  auto probs = outputs[0].toTensor();
  auto r_probs = outputs[1].toTensor();
  CHECK_EQ(probs.size(0), num_hyps);
  CHECK_EQ(probs.size(1), max_hyps_len);
  // Step 3: Compute rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    float score = 0.0f;
    // left to right decoder score
    score = AttentionDecoderScore(probs[i], hyp, eos);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (opts_.reverse_weight > 0) {
      // Right to left score
      CHECK_EQ(r_probs.size(0), num_hyps);
      CHECK_EQ(r_probs.size(1), max_hyps_len);
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = AttentionDecoderScore(r_probs[i], r_hyp, eos);
    }
    // combined reverse attention score
    score =
        (score * (1 - opts_.reverse_weight)) + (r_score * opts_.reverse_weight);
    // combined ctc score
    result_[i].score =
        opts_.rescoring_weight * score + opts_.ctc_weight * result_[i].score;
  }
  std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
}

}  // namespace wenet
