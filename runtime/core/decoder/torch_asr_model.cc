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

#include "decoder/torch_asr_model.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include "torch/script.h"
#ifndef IOS
#include "torch/torch.h"
#endif
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

namespace wenet {

#ifndef IOS
void TorchAsrModel::InitEngineThreads(int num_threads) {
  // For multi-thread performance
  at::set_num_threads(num_threads);
  VLOG(1) << "Num intra-op threads: " << at::get_num_threads();
}
#endif

void TorchAsrModel::Read(const std::string& model_path) {
  torch::DeviceType device = at::kCPU;
#ifdef USE_GPU
  if (!torch::cuda::is_available()) {
    VLOG(1) << "CUDA is not available! Please check your GPU settings";
    throw std::runtime_error("CUDA is not available!");
  } else {
    VLOG(1) << "CUDA available! Running on GPU";
    device = at::kCUDA;
  }
#endif
#ifdef USE_IPEX
  torch::jit::setTensorExprFuserEnabled(false);
#endif
  torch::jit::script::Module model = torch::jit::load(model_path, device);
  model_ = std::make_shared<TorchModule>(std::move(model));
  torch::NoGradGuard no_grad;
  model_->eval();
  torch::jit::IValue o1 = model_->run_method("subsampling_rate");
  CHECK_EQ(o1.isInt(), true);
  subsampling_rate_ = o1.toInt();
  torch::jit::IValue o2 = model_->run_method("right_context");
  CHECK_EQ(o2.isInt(), true);
  right_context_ = o2.toInt();
  torch::jit::IValue o3 = model_->run_method("sos_symbol");
  CHECK_EQ(o3.isInt(), true);
  sos_ = o3.toInt();
  torch::jit::IValue o4 = model_->run_method("eos_symbol");
  CHECK_EQ(o4.isInt(), true);
  eos_ = o4.toInt();
  torch::jit::IValue o5 = model_->run_method("is_bidirectional_decoder");
  CHECK_EQ(o5.isBool(), true);
  is_bidirectional_decoder_ = o5.toBool();

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy static0 = {
      {torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(static0);

  VLOG(1) << "Torch Model Info:";
  VLOG(1) << "\tsubsampling_rate " << subsampling_rate_;
  VLOG(1) << "\tright context " << right_context_;
  VLOG(1) << "\tsos " << sos_;
  VLOG(1) << "\teos " << eos_;
  VLOG(1) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
}

TorchAsrModel::TorchAsrModel(const TorchAsrModel& other) {
  // 1. Init the model info
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;
  // 2. Model copy, just copy the model ptr since:
  // PyTorch allows using multiple CPU threads during TorchScript model
  // inference, please see https://pytorch.org/docs/stable/notes/cpu_
  // threading_torchscript_inference.html
  model_ = other.model_;

  // NOTE(Binbin Zhang):
  // inner states for forward are not copied here.
}

std::shared_ptr<AsrModel> TorchAsrModel::Copy() const {
  auto asr_model = std::make_shared<TorchAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

void TorchAsrModel::Reset() {
  offset_ = 0;
  att_cache_ = std::move(torch::zeros({0, 0, 0, 0}));
  cnn_cache_ = std::move(torch::zeros({0, 0, 0, 0}));
  encoder_outs_.clear();
  cached_feature_.clear();
}

void TorchAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  // 1. Prepare libtorch required data, splice cached_feature_ and chunk_feats
  // The first dimension is for batchsize, which is 1.
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  torch::Tensor feats =
      torch::zeros({1, num_frames, feature_dim}, torch::kFloat);
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    torch::Tensor row =
        torch::from_blob(const_cast<float*>(cached_feature_[i].data()),
                         {feature_dim}, torch::kFloat)
            .clone();
    feats[0][i] = std::move(row);
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    torch::Tensor row =
        torch::from_blob(const_cast<float*>(chunk_feats[i].data()),
                         {feature_dim}, torch::kFloat)
            .clone();
    feats[0][cached_feature_.size() + i] = std::move(row);
  }

  // 2. Encoder chunk forward
#ifdef USE_GPU
  feats = feats.to(at::kCUDA);
  att_cache_ = att_cache_.to(at::kCUDA);
  cnn_cache_ = cnn_cache_.to(at::kCUDA);
#endif
  int required_cache_size = chunk_size_ * num_left_chunks_;
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs = {feats, offset_, required_cache_size,
                                            att_cache_, cnn_cache_};

  // Refer interfaces in wenet/transformer/asr_model.py
  auto outputs =
      model_->get_method("forward_encoder_chunk")(inputs).toTuple()->elements();
  CHECK_EQ(outputs.size(), 3);
#ifdef USE_GPU
  torch::Tensor chunk_out = outputs[0].toTensor().to(at::kCPU);
  att_cache_ = outputs[1].toTensor().to(at::kCPU);
  cnn_cache_ = outputs[2].toTensor().to(at::kCPU);
#else
  torch::Tensor chunk_out = outputs[0].toTensor();
  att_cache_ = outputs[1].toTensor();
  cnn_cache_ = outputs[2].toTensor();
#endif
  offset_ += chunk_out.size(1);

  // The first dimension of returned value is for batchsize, which is 1
#ifdef USE_GPU
  chunk_out = chunk_out.to(at::kCUDA);
  torch::Tensor ctc_log_probs =
      model_->run_method("ctc_activation", chunk_out).toTensor();
  ctc_log_probs = ctc_log_probs.to(at::kCPU)[0];
  encoder_outs_.push_back(std::move(chunk_out.to(at::kCPU)));
#else
  torch::Tensor ctc_log_probs =
      model_->run_method("ctc_activation", chunk_out).toTensor()[0];
  encoder_outs_.push_back(std::move(chunk_out));
#endif

  // Copy to output
  int num_outputs = ctc_log_probs.size(0);
  int output_dim = ctc_log_probs.size(1);
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), ctc_log_probs[i].data_ptr(),
           sizeof(float) * output_dim);
  }
}

float TorchAsrModel::ComputeAttentionScore(const torch::Tensor& prob,
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

void TorchAsrModel::AttentionRescoring(
    const std::vector<std::vector<int>>& hyps, float reverse_weight,
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

  torch::NoGradGuard no_grad;
  // Step 1: Prepare input for libtorch
  torch::Tensor hyps_length = torch::zeros({num_hyps}, torch::kLong);
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_length[i] = static_cast<int64_t>(length);
  }
  torch::Tensor hyps_tensor =
      torch::zeros({num_hyps, max_hyps_len}, torch::kLong);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_tensor[i][0] = sos_;
    for (size_t j = 0; j < hyp.size(); ++j) {
      hyps_tensor[i][j + 1] = hyp[j];
    }
  }

  // Step 2: Forward attention decoder by hyps and corresponding encoder_outs_
  torch::Tensor encoder_out = torch::cat(encoder_outs_, 1);
#ifdef USE_GPU
  hyps_tensor = hyps_tensor.to(at::kCUDA);
  hyps_length = hyps_length.to(at::kCUDA);
  encoder_out = encoder_out.to(at::kCUDA);
#endif
  auto outputs = model_
                     ->run_method("forward_attention_decoder", hyps_tensor,
                                  hyps_length, encoder_out, reverse_weight)
                     .toTuple()
                     ->elements();
#ifdef USE_GPU
  auto probs = outputs[0].toTensor().to(at::kCPU);
  auto r_probs = outputs[1].toTensor().to(at::kCPU);
#else
  auto probs = outputs[0].toTensor();
  auto r_probs = outputs[1].toTensor();
#endif
  CHECK_EQ(probs.size(0), num_hyps);
  CHECK_EQ(probs.size(1), max_hyps_len);

  // Step 3: Compute rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left-to-right decoder score
    score = ComputeAttentionScore(probs[i], hyp, eos_);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      // right-to-left score
      CHECK_EQ(r_probs.size(0), num_hyps);
      CHECK_EQ(r_probs.size(1), max_hyps_len);
      std::vector<int> r_hyp(hyp.size());
      std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
      // right to left decoder score
      r_score = ComputeAttentionScore(r_probs[i], r_hyp, eos_);
    }

    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =
        score * (1 - reverse_weight) + r_score * reverse_weight;
  }
}

}  // namespace wenet
