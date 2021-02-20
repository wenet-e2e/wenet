// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/torch_asr_decoder.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace wenet {

TorchAsrDecoder::TorchAsrDecoder(
    std::shared_ptr<FeaturePipeline> feature_pipeline,
    std::shared_ptr<TorchAsrModel> model, const SymbolTable& symbol_table,
    const DecodeOptions& opts)
    : feature_pipeline_(feature_pipeline),
      model_(model),
      symbol_table_(symbol_table),
      opts_(opts),
      ctc_prefix_beam_searcher_(new CtcPrefixBeamSearch(opts.ctc_search_opts)) {
}

void TorchAsrDecoder::Reset() {
  start_ = false;
  result_ = "";
  offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  subsampling_cache_ = std::move(torch::jit::IValue());
  elayers_output_cache_ = std::move(torch::jit::IValue());
  conformer_cnn_cache_ = std::move(torch::jit::IValue());
  encoder_outs_.clear();
  cached_feature_.clear();
  ctc_prefix_beam_searcher_->Reset();
  feature_pipeline_->Reset();
}

bool TorchAsrDecoder::Decode() {
  bool finish = this->AdvanceDecoding();
  if (finish) {
    // Do attention rescoring
    auto start = std::chrono::steady_clock::now();
    AttentionRescoring();
    auto end = std::chrono::steady_clock::now();
    LOG(INFO) << "Rescoring cost latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms.";
    return true;
  }
  return false;
}

bool TorchAsrDecoder::AdvanceDecoding() {
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
  bool finish = !feature_pipeline_->Read(num_requried_frames, &chunk_feats);
  num_frames_in_current_chunk_ = chunk_feats.size();
  LOG(INFO) << "Required " << num_requried_frames << " get "
            << chunk_feats.size();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  // The total frames should be big enough to get just one output
  if (num_frames >= right_context + 1) {
    // 1. Prepare libtorch requried data, splice cached_feature_ and chunk_feats
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

    // 2. Encoder chunk forward
    int requried_cache_size = opts_.chunk_size * opts_.num_left_chunks;
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {feats,
                                              offset_,
                                              requried_cache_size,
                                              subsampling_cache_,
                                              elayers_output_cache_,
                                              conformer_cnn_cache_};
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
    // The first dimension is a fake dimension, it's 1 for one utterance,
    // so just ignore it here.
    torch::Tensor ctc_log_probs = model_->torch_model()
                                      ->run_method("ctc_activation", chunk_out)
                                      .toTensor()[0];
    encoder_outs_.push_back(std::move(chunk_out));
    ctc_prefix_beam_searcher_->Search(ctc_log_probs);
    auto hypotheses = ctc_prefix_beam_searcher_->hypotheses();
    const std::vector<int>& best_hyp = hypotheses[0];
    result_ = "";
    for (size_t i = 0; i < best_hyp.size(); ++i) {
      result_ += symbol_table_.Find(best_hyp[i]);
    }
    VLOG(1) << "Partial CTC result " << result_;

    // 3. cache feature for next chunk
    if (!finish) {
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
  return finish;
}

static bool CompareFunc(const std::pair<int, float>& a,
                        const std::pair<int, float>& b) {
  return a.second > b.second;
}

void TorchAsrDecoder::AttentionRescoring() {
  int sos = model_->sos();
  int eos = model_->eos();
  auto hypotheses = ctc_prefix_beam_searcher_->hypotheses();
  int num_hyps = hypotheses.size();
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

  // Step 2: forward attention decoder by hyps and corresponding encoder_outs_
  torch::Tensor encoder_out = torch::cat(encoder_outs_, 1);
  torch::Tensor probs = model_->torch_model()
                            ->run_method("forward_attention_decoder",
                                         hyps_tensor, hyps_length, encoder_out)
                            .toTensor();
  CHECK_EQ(probs.size(0), num_hyps);
  CHECK_EQ(probs.size(1), max_hyps_len);

  // Step 3: Compute rescoring score
  // (id, score) pair for later sort
  std::vector<std::pair<int, float>> weighted_scores(num_hyps);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    float score = 0.0f;
    for (size_t j = 0; j < hyp.size(); ++j) {
      score += probs[i][j][hyp[j]].item<float>();
    }
    score += probs[i][hyp.size()][eos].item<float>();
    // TODO(Binbin Zhang): Combine CTC and attention decoder score
    weighted_scores[i].first = i;
    weighted_scores[i].second = score;
  }

  std::sort(weighted_scores.begin(), weighted_scores.end(), CompareFunc);
  for (size_t i = 0; i < weighted_scores.size(); ++i) {
    std::string result;
    int best_k = weighted_scores[i].first;
    for (size_t j = 0; j < hypotheses[best_k].size(); ++j) {
      result += symbol_table_.Find(hypotheses[best_k][j]);
    }
    VLOG(1) << "ctc index " << best_k << " result " << result << " score "
            << weighted_scores[i].second;
    if (0 == i) {
      result_ = result;
    }
  }
}

}  // namespace wenet
