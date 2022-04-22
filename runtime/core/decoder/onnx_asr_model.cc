// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
// Copyright 2022 Mobvoi Inc. All Rights Reserved.
// Author: lizexuan@huya.com

#include "decoder/onnx_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace wenet {

void OnnxAsrModel::Read(const std::string& model_dir, const int num_threads) {
  std::string encoder_onnx_path = model_dir + "/encoder.onnx";
  std::string rescore_onnx_path = model_dir + "/decoder.onnx";
  std::string ctc_onnx_path = model_dir + "/ctc.onnx";

  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetInterOpNumThreads(num_threads);

    Ort::Session encoder_session{env_, encoder_onnx_path.data(),
                                 session_options};
    encoder_session_ =
        std::make_shared<Ort::Session>(std::move(encoder_session));

    Ort::Session rescore_session{env_, rescore_onnx_path.data(),
                                 session_options};
    rescore_session_ =
        std::make_shared<Ort::Session>(std::move(rescore_session));

    Ort::Session ctc_session{env_, ctc_onnx_path.data(), session_options};
    ctc_session_ = std::make_shared<Ort::Session>(std::move(ctc_session));

  } catch (std::exception const& e) {
    LOG(FATAL) << "error when load onnx model";
    exit(0);
  }

  auto model_metadata = encoder_session_->GetModelMetadata();
  int64_t num_keys;
  OrtAllocator* allocator;

  OrtStatus* status = Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator);

  auto output_size =
      model_metadata.LookupCustomMetadataMap("output_size", allocator);
  auto num_blocks =
      model_metadata.LookupCustomMetadataMap("num_blocks", allocator);
  auto cnn_module_kernel =
      model_metadata.LookupCustomMetadataMap("cnn_module_kernel", allocator);
  auto subsampling_rate =
      model_metadata.LookupCustomMetadataMap("subsampling_rate", allocator);
  auto right_context =
      model_metadata.LookupCustomMetadataMap("right_context", allocator);
  auto sos_symbol =
      model_metadata.LookupCustomMetadataMap("sos_symbol", allocator);
  auto eos_symbol =
      model_metadata.LookupCustomMetadataMap("eos_symbol", allocator);
  auto is_bidirectional_decoder = model_metadata.LookupCustomMetadataMap(
      "is_bidirectional_decoder", allocator);
  auto chunk_size =
      model_metadata.LookupCustomMetadataMap("chunk_size", allocator);
  auto left_chunks =
      model_metadata.LookupCustomMetadataMap("left_chunks", allocator);

  encoder_output_size_ = atoi(output_size);
  num_blocks_ = atoi(num_blocks);
  cnn_module_kernel_ = atoi(cnn_module_kernel);
  subsampling_rate_ = atoi(subsampling_rate);
  right_context_ = atoi(right_context);
  sos_ = atoi(sos_symbol);
  eos_ = atoi(eos_symbol);
  is_bidirectional_decoder_ = atoi(is_bidirectional_decoder);
  chunk_size_ = atoi(chunk_size);
  num_left_chunks_ = atoi(left_chunks);

  LOG(INFO) << "Onnx Model Info:";
  LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
  LOG(INFO) << "\tnum_blocks " << num_blocks_;
  LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;
}

OnnxAsrModel::OnnxAsrModel(const OnnxAsrModel& other) {
  encoder_output_size_ = other.encoder_output_size_;
  num_blocks_ = other.num_blocks_;
  cnn_module_kernel_ = other.cnn_module_kernel_;
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  encoder_session_ = other.encoder_session_;
  rescore_session_ = other.rescore_session_;
  ctc_session_ = other.ctc_session_;
}

std::shared_ptr<AsrModel> OnnxAsrModel::Copy() const {
  auto asr_model = std::make_shared<OnnxAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

void OnnxAsrModel::Reset() {
  offset_ = 0;
  int required_cache_size;
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;
    att_cache_.resize(num_blocks_ * 4 * required_cache_size *
                      encoder_output_size_ / 4 * 2);
    const int64_t att_cache_shape[] = {num_blocks_, 4, required_cache_size,
                                       encoder_output_size_ / 4 * 2};
    att_cache_ort_ = Ort::Value::CreateTensor<float>(
        memory_info_, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);

    att_mask_.resize(required_cache_size + chunk_size_);
    for (size_t i = 0; i < required_cache_size + chunk_size_; ++i) {
      att_mask_[i] = i >= required_cache_size;
    }
    const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
    att_mask_ort_ = Ort::Value::CreateTensor<bool>(
        memory_info_, reinterpret_cast<bool*>(att_mask_.data()),
        att_mask_.size(), att_mask_shape, 3);
  } else {
    att_cache_.resize(0);
    const int64_t att_cache_shape[] = {num_blocks_, 4, 0,
                                       encoder_output_size_ / 4 * 2};
    att_cache_ort_ = Ort::Value::CreateTensor<float>(
        memory_info_, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);

    att_mask_.resize(0);
    const int64_t att_mask_shape[] = {0, 0, 0};
    att_mask_ort_ = Ort::Value::CreateTensor<bool>(
        memory_info_, reinterpret_cast<bool*>(att_mask_.data()),
        att_mask_.size(), att_mask_shape, 3);
  }

  cnn_cache_.resize(num_blocks_ * encoder_output_size_ *
                    (cnn_module_kernel_ - 1));
  const int64_t cnn_cache_shape[] = {num_blocks_, 1, encoder_output_size_,
                                     cnn_module_kernel_ - 1};

  cnn_cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info_, cnn_cache_.data(), cnn_cache_.size(), cnn_cache_shape, 4);

  encoder_outs_.clear();
}

void OnnxAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  int requried_cache_size = chunk_size_ * num_left_chunks_;
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
  const int64_t input_shape[3] = {1, num_frames, feature_dim};

  std::vector<int64_t> offset{offset_};
  const int64_t offset_shape[1] = {1};

  std::vector<int64_t> requried_cache_size_vec{requried_cache_size};
  const int64_t requried_cache_size_shape[1] = {1};

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
  tensors.emplace_back(std::move(att_cache_ort_));
  tensors.emplace_back(std::move(cnn_cache_ort_));
  tensors.emplace_back(std::move(att_mask_ort_));

  std::vector<Ort::Value> ort_outputs =
      encoder_session_->Run(Ort::RunOptions{nullptr}, input_names_,
                            tensors.data(), 6, output_names_, 3);

  offset_ += ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  att_cache_ort_ = std::move(ort_outputs[1]);
  cnn_cache_ort_ = std::move(ort_outputs[2]);

  std::vector<Ort::Value> ctc_tensors;
  ctc_tensors.emplace_back(std::move(ort_outputs[0]));

  std::vector<Ort::Value> ctc_ort_outputs =
      ctc_session_->Run(Ort::RunOptions{nullptr}, ctc_input_names_,
                        ctc_tensors.data(), 1, ctc_output_names_, 1);
  encoder_outs_.push_back(std::move(ctc_tensors[0]));

  float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
           sizeof(float) * output_dim);
  }
}

float OnnxAsrModel::ComputeAttentionScore(const float* prob,
                                          const std::vector<int>& hyp, int eos,
                                          int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void OnnxAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                      float reverse_weight,
                                      std::vector<float>* rescoring_score) {
  CHECK(rescoring_score != nullptr);
  int num_hyps = hyps.size();
  rescoring_score->resize(num_hyps, 0.0f);

  if (num_hyps == 0) {
    return;
  }
  // No encoder output
  if (encoder_outs_.size() == 0) {
    return;
  }

  std::vector<int64_t> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(static_cast<int64_t>(length));
  }

  std::vector<float> rescore_input;
  int encoder_len = 0;
  for (int i = 0; i < encoder_outs_.size(); i++) {
    float* encoder_outs_data = encoder_outs_[i].GetTensorMutableData<float>();
    auto type_info = encoder_outs_[i].GetTensorTypeAndShapeInfo();
    for (int j = 0; j < type_info.GetElementCount(); j++) {
      rescore_input.emplace_back(encoder_outs_data[j]);
    }
    encoder_len += type_info.GetShape()[1];
  }

  const int64_t decode_input_shape[] = {1, encoder_len, encoder_output_size_};

  std::vector<int64_t> hyps_pad;

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_pad.emplace_back(sos_);
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

  std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
      Ort::RunOptions{nullptr}, decode_input_names_, rescore_tensors.data(),
      rescore_tensors.size(), decode_output_names_, 2);

  float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
  float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();

  auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
  int decode_out_len = type_info.GetShape()[2];

  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left to right decoder score
    score = ComputeAttentionScore(
        decoder_outs_data + max_hyps_len * decode_out_len * i, hyp, eos_,
        decode_out_len);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = ComputeAttentionScore(
          r_decoder_outs_data + max_hyps_len * decode_out_len * i, r_hyp, eos_,
          decode_out_len);
    }
    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =
        score * (1 - reverse_weight) + r_score * reverse_weight;
  }
}

}  // namespace wenet
