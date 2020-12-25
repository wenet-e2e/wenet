// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <limits>
#include <utility>

#include "decoder/torch_asr_decoder.h"

namespace wenet {

TorchAsrDecoder::TorchAsrDecoder(
    std::shared_ptr<FeaturePipeline> feature_pipeline,
    std::shared_ptr<TorchAsrModel> model,
    const SymbolTable& symbol_table,
    const DecodeOptions& opts)
    : feature_pipeline_(feature_pipeline),
      model_(model),
      symbol_table_(symbol_table),
      opts_(opts),
      ctc_prefix_beam_searcher_(new CtcPrefixBeamSearch(opts.ctc_search_opts))
    {}

void TorchAsrDecoder::Reset() {
  start_ = false;
  cached_feature_.clear();
}

bool TorchAsrDecoder::Decode() {
  bool finish = this->AdvanceDecoding();
  if (finish) {
    // Do attention rescoring
    AttentionRescoring();
    return true;
  } else {
    return false;
  }
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
    if (!start_) {  // First batch
      int context = right_context + 1;  // Add current frame
      num_requried_frames = (opts_.chunk_size - 1) *
                            subsampling_rate + context;
    } else {
      num_requried_frames = opts_.chunk_size * subsampling_rate;
    }
  } else {
    num_requried_frames = std::numeric_limits<int>::max();
  }
  std::vector<std::vector<float> > chunk_feats;
  // If not okay, that means we reach the end of the input
  bool finish = !feature_pipeline_->Read(num_requried_frames, &chunk_feats);
  LOG(INFO) << "Required " << num_requried_frames
            << " get " << chunk_feats.size();
  int num_frames = cached_feature_.size() + chunk_feats.size();
  // The total frames should be big enough to get just one output
  if (num_frames >= right_context + 1) {
    // 1. Prepare libtorch requried data, splice cached_feature_ and chunk_feats
    torch::Tensor feats = torch::zeros(
        {1, num_frames, feature_dim}, torch::kFloat);
    for (size_t i = 0; i < cached_feature_.size(); i++) {
      torch::Tensor row = torch::from_blob(cached_feature_[i].data(),
          {feature_dim}, torch::kFloat).clone();
      feats[0][i] = std::move(row);
    }
    int offset = cached_feature_.size();
    for (size_t i = 0; i < chunk_feats.size(); ++i) {
      torch::Tensor row = torch::from_blob(chunk_feats[i].data(),
          {feature_dim}, torch::kFloat).clone();
      feats[0][offset + i] = std::move(row);
    }

    // 2. Encoder chunk forward
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {feats, subsampling_cache_,
        elayers_output_cache_, conformer_cnn_cache_};
    auto outputs = model_->torch_model()->get_method(
        "forward_encoder_chunk")(inputs).toTuple()->elements();
    CHECK_EQ(outputs.size(), 4);
    // The encoder_out_ is from time 0 to current chunk, and offset_ is the
    // offset of current chunk, so to get output of current chunk, just slice
    // as the following code
    encoder_out_ = outputs[0].toTensor();
    subsampling_cache_ = outputs[1];
    elayers_output_cache_ = outputs[2];
    conformer_cnn_cache_ = outputs[3];
    torch::Tensor chunk_out = encoder_out_.slice(1, offset_,
        encoder_out_.size(1));
    offset_ = encoder_out_.size(1);
    // The first dimension is a fake dimension, it's 1 for one utterance,
    // so just ignore it here.
    torch::Tensor ctc_log_probs = model_->torch_model()->run_method(
        "ctc_activation", chunk_out).toTensor()[0];
    ctc_prefix_beam_searcher_->Search(ctc_log_probs);
    auto hypotheses = ctc_prefix_beam_searcher_->hypotheses();
    const std::vector<int>& best_hyp = hypotheses[0];
    hyp_ = "";
    for (size_t i = 0; i < best_hyp.size(); i++) {
      hyp_ += symbol_table_.Find(best_hyp[i]);
    }
    LOG(INFO) << "Partial CTC result " << hyp_;

    // 3. cache feature for next chunk
    if (!finish) {
      // TODO(Binbin Zhang): Only deal the case when
      // chunk_feats.size() > cached_feature_size_ here, and it's consistent
      // with our current model, refine it later if we have new model or
      // new requirements
      CHECK(chunk_feats.size() >= cached_feature_size);
      cached_feature_.resize(cached_feature_size);
      for (int i = 0; i < cached_feature_size; i++) {
        cached_feature_[i] = std::move(
            chunk_feats[chunk_feats.size() - cached_feature_size + i]);
      }
    }
  }

  start_ = true;
  return finish;
}

void TorchAsrDecoder::AttentionRescoring() {
}

}  // namespace wenet
