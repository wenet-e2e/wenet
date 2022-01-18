// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
// Copyright 2021 Huya Inc. All Rights Reserved.
// Author: lizexuan@huya.com (Zexuan Li)

#include "decoder/onnx_asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "decoder/ctc_endpoint.h"
#include "utils/timer.h"

namespace wenet {

OnnxAsrDecoder::OnnxAsrDecoder(
    std::shared_ptr<FeaturePipeline> feature_pipeline,
    std::shared_ptr<DecodeResource> resource, const DecodeOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      model_(resource->onnx_model),
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
  OnnxReset();
}

void OnnxAsrDecoder::Reset() {
  start_ = false;
  result_.clear();
  offset_ = 1;
  num_frames_ = 0;
  global_frame_offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  OnnxReset();
  cached_feature_.clear();
  searcher_->Reset();
  feature_pipeline_->Reset();
  ctc_endpointer_->Reset();
}

void OnnxAsrDecoder::ResetContinuousDecoding() {
  global_frame_offset_ = num_frames_;
  start_ = false;
  result_.clear();
  offset_ = 0;
  num_frames_in_current_chunk_ = 0;
  OnnxReset();
  cached_feature_.clear();
  searcher_->Reset();
  ctc_endpointer_->Reset();
}

void OnnxAsrDecoder::OnnxReset() {
  int encoder_output_size = model_->encoder_output_size();
  int num_blocks = model_->num_blocks();
  int cnn_module_kernel = model_->cnn_module_kernel();

  subsampling_cache_.resize(encoder_output_size);
  const int64_t subsampling_cache_shape[] = {1, 1, encoder_output_size};

  elayers_output_cache_.resize(encoder_output_size * num_blocks);
  const int64_t elayers_output_cache_shape[] = {num_blocks, 1, 1,
                                                encoder_output_size};

  conformer_cnn_cache_.resize(num_blocks * encoder_output_size *
                              cnn_module_kernel);
  const int64_t conformer_cnn_cache_shape[] = {
      num_blocks, 1, encoder_output_size, cnn_module_kernel};

  subsampling_cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info_, subsampling_cache_.data(), subsampling_cache_.size(),
      subsampling_cache_shape, 3);
  elayers_output_cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info_, elayers_output_cache_.data(), elayers_output_cache_.size(),
      elayers_output_cache_shape, 4);
  conformer_cnn_cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info_, conformer_cnn_cache_.data(), conformer_cnn_cache_.size(),
      conformer_cnn_cache_shape, 4);

  encoder_outs_ort_.clear();
}

DecodeState OnnxAsrDecoder::Decode() { return this->AdvanceDecoding(); }

void OnnxAsrDecoder::Rescoring() {
  // Do attention rescoring
  Timer timer;
  AttentionRescoring();
  LOG(INFO) << "Rescoring cost latency: " << timer.Elapsed() << "ms.";
}

DecodeState OnnxAsrDecoder::AdvanceDecoding() {
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
    std::vector<float> model_input;
    for (size_t i = 0; i < cached_feature_.size(); ++i) {
      for (int j = 0; j < feature_dim; j++) {
        model_input.emplace_back(cached_feature_[i][j]);
      }
    }
    for (size_t i = 0; i < chunk_feats.size(); ++i) {
      for (int j = 0; j < feature_dim; j++) {
        model_input.emplace_back(chunk_feats[i][j]);
      }
    }

    Timer timer;
    std::vector<int64_t> offset{offset_};
    const int64_t offset_shape[1] = {1};
    int requried_cache_size = opts_.chunk_size * opts_.num_left_chunks;
    std::vector<int64_t> requried_cache_size_vec{requried_cache_size};
    const int64_t requried_cache_size_shape[1] = {1};

    const int64_t input_shape[3] = {1, num_frames, feature_dim};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, model_input.data(), model_input.size(), input_shape, 3);
    Ort::Value offset_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_, offset.data(), 1, offset_shape, 1);
    Ort::Value requried_cache_size_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_, requried_cache_size_vec.data(), 1,
        requried_cache_size_shape, 1);
    std::vector<Ort::Value> tensors;
    tensors.emplace_back(std::move(input_tensor));
    tensors.emplace_back(std::move(offset_tensor));
    tensors.emplace_back(std::move(requried_cache_size_tensor));
    tensors.emplace_back(std::move(subsampling_cache_ort_));
    tensors.emplace_back(std::move(elayers_output_cache_ort_));
    tensors.emplace_back(std::move(conformer_cnn_cache_ort_));

    std::vector<Ort::Value> ort_outputs =
        model_->encoder_session()->Run(Ort::RunOptions{nullptr}, input_names_,
                                       tensors.data(), 6, output_names_, 4);

    offset_ += ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    subsampling_cache_ort_ = std::move(ort_outputs[1]);
    elayers_output_cache_ort_ = std::move(ort_outputs[2]);
    conformer_cnn_cache_ort_ = std::move(ort_outputs[3]);

    std::vector<Ort::Value> ctc_tensors;
    ctc_tensors.emplace_back(std::move(ort_outputs[0]));

    std::vector<Ort::Value> ctc_ort_outputs =
        model_->ctc_session()->Run(Ort::RunOptions{nullptr}, ctc_input_names_,
                                   ctc_tensors.data(), 1, ctc_output_names_, 1);
    encoder_outs_ort_.push_back(std::move(ctc_tensors[0]));

    float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
    auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

    int num_subsampling_frame = type_info.GetShape()[1];
    int num_probability = type_info.GetShape()[2];

    std::vector<std::vector<float>> ctc_log_probs_vec;

    for (size_t t = 0; t < num_subsampling_frame; ++t) {
      std::vector<float> logp_t(
          logp_data + t * num_probability,
          logp_data + t * num_probability + num_probability);
      ctc_log_probs_vec.emplace_back(logp_t);
    }

    torch::Tensor ctc_log_probs =
        torch::zeros({num_subsampling_frame, num_probability}, torch::kFloat);

    for (size_t t = 0; t < num_subsampling_frame; ++t) {
      torch::Tensor row = torch::from_blob(ctc_log_probs_vec[t].data(),
                                           {num_probability}, torch::kFloat);
      ctc_log_probs[t] = std::move(row);
    }

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

void OnnxAsrDecoder::UpdateResult(bool finish) {
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
        int start = j > 0 ? ((time_stamp[j - 1] + time_stamp[j]) / 2 *
                             frame_shift_in_ms())
                          : 0;
        int end = j < input.size() - 1 ? ((time_stamp[j] + time_stamp[j + 1]) /
                                          2 * frame_shift_in_ms())
                                       : offset_ * frame_shift_in_ms();
        WordPiece word_piece(word, offset + start, offset + end);
        path.word_pieces.emplace_back(word_piece);
      }
    }
    path.sentence = post_processor_->Process(path.sentence, finish);
    result_.emplace_back(path);
  }

  if (DecodedSomething()) {
    VLOG(1) << "Partial CTC result " << result_[0].sentence;
  }
}

float OnnxAsrDecoder::AttentionDecoderScore(const float* prob,
                                            const std::vector<int>& hyp,
                                            int eos, int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void OnnxAsrDecoder::AttentionRescoring() {
  searcher_->FinalizeSearch();
  UpdateResult(true);
  // No need to do rescoring
  if (0.0 == opts_.rescoring_weight) {
    return;
  }

  // No encoder output
  if (encoder_outs_ort_.size() == 0) {
    return;
  }

  int sos = model_->sos();
  int eos = model_->eos();

  const auto& hypotheses = searcher_->Inputs();

  int num_hyps = hypotheses.size();
  if (num_hyps <= 0) {
    return;
  }

  std::vector<int64_t> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hypotheses[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(static_cast<int64_t>(length));
  }

  std::vector<float> rescore_input;
  int encoder_len = 0;
  for (int i = 0; i < encoder_outs_ort_.size(); i++) {
    float* encoder_outs_data =
        encoder_outs_ort_[i].GetTensorMutableData<float>();
    auto type_info = encoder_outs_ort_[i].GetTensorTypeAndShapeInfo();
    for (int j = 0; j < type_info.GetElementCount(); j++) {
      rescore_input.emplace_back(encoder_outs_data[j]);
    }
    encoder_len += type_info.GetShape()[1];
  }

  int encoder_output_size = model_->encoder_output_size();
  const int64_t decode_input_shape[] = {1, encoder_len, encoder_output_size};

  std::vector<int64_t> hyps_pad;

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    hyps_pad.emplace_back(sos);
    size_t j = 0;
    for (; j < hyp.size(); ++j) {
      hyps_pad.emplace_back(hyp[j]);
    }
    if (j == max_hyps_len - 1) {
      continue;
    }
    for (; j < max_hyps_len - 1; ++j) {
      hyps_pad.emplace_back(0);
    }
  }

  const int64_t hyps_pad_shape[] = {num_hyps, max_hyps_len};

  const int64_t hyps_lens_shape[] = {num_hyps};

  Ort::Value decode_input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info_, rescore_input.data(), rescore_input.size(),
      decode_input_shape, 3);
  Ort::Value hyps_pad_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info_, hyps_pad.data(), hyps_pad.size(), hyps_pad_shape, 2);
  Ort::Value hyps_lens_tensor_ = Ort::Value::CreateTensor<int64_t>(
      memory_info_, hyps_lens.data(), hyps_lens.size(), hyps_lens_shape, 1);

  std::vector<Ort::Value> rescore_tensors;

  rescore_tensors.emplace_back(std::move(hyps_pad_tensor_));
  rescore_tensors.emplace_back(std::move(hyps_lens_tensor_));
  rescore_tensors.emplace_back(std::move(decode_input_tensor_));

  std::vector<Ort::Value> rescore_outputs = model_->rescore_session()->Run(
      Ort::RunOptions{nullptr}, decode_input_names_, rescore_tensors.data(),
      rescore_tensors.size(), decode_output_names_, 2);

  float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
  float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();

  auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
  int decode_out_len = type_info.GetShape()[2];

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hypotheses[i];
    float score = 0.0f;
    // left to right decoder score
    score = AttentionDecoderScore(
        decoder_outs_data + max_hyps_len * decode_out_len * i, hyp, eos,
        decode_out_len);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (opts_.reverse_weight > 0) {
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = AttentionDecoderScore(
          r_decoder_outs_data + max_hyps_len * decode_out_len * i, r_hyp, eos,
          decode_out_len);
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
