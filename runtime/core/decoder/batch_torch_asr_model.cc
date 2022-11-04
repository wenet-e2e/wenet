// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
//               2022 SoundDataConverge Co.LTD (Weiliang Chong)
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


#include "decoder/batch_torch_asr_model.h"

#ifdef USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <algorithm>
#include <memory>
#include <utility>
#include <stdexcept>

#include "torch/script.h"
#include "torch/torch.h"

namespace wenet {

void BatchTorchAsrModel::InitEngineThreads(int num_threads) {
  VLOG(1) << "Num intra-op default threads: " << at::get_num_threads();
  // For multi-thread performance
  at::set_num_threads(num_threads);
  // Note: Do not call the set_num_interop_threads function more than once.
  // Please see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/
  // ParallelThreadPoolNative.cpp#L54-L56
  at::set_num_interop_threads(1);
  VLOG(1) << "Num intra-op threads: " << at::get_num_threads();
  VLOG(1) << "Num inter-op threads: " << at::get_num_interop_threads();
}

void BatchTorchAsrModel::Read(const std::string& model_path) {
#ifdef USE_GPU
  if (!torch::cuda::is_available()) {
    VLOG(1) << "CUDA is not available! Please check your GPU settings";
    throw std::runtime_error("CUDA is not available!");
  } else {
    VLOG(1) << "CUDA is available! Running on GPU";
    device_ = at::kCUDA;
  }
#endif
  torch::jit::script::Module model = torch::jit::load(model_path, device_);
  model_ = std::make_shared<TorchModule>(std::move(model));
  torch::NoGradGuard no_grad;
  model_->eval();
  torch::jit::IValue o1 = model_->run_method("subsampling_rate");
  CHECK_EQ(o1.isInt(), true);
  subsampling_rate_ = o1.toInt();
  torch::jit::IValue o2 = model_->run_method("right_context");
  CHECK_EQ(o2.isInt(), true);
  torch::jit::IValue o3 = model_->run_method("sos_symbol");
  CHECK_EQ(o3.isInt(), true);
  sos_ = o3.toInt();
  torch::jit::IValue o4 = model_->run_method("eos_symbol");
  CHECK_EQ(o4.isInt(), true);
  eos_ = o4.toInt();
  torch::jit::IValue o5 = model_->run_method("is_bidirectional_decoder");
  CHECK_EQ(o5.isBool(), true);
  is_bidirectional_decoder_ = o5.toBool();

  VLOG(1) << "Torch Model Info:";
  VLOG(1) << "\tsubsampling_rate " << subsampling_rate_;
  VLOG(1) << "\tsos " << sos_;
  VLOG(1) << "\teos " << eos_;
  VLOG(1) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
}

BatchTorchAsrModel::BatchTorchAsrModel(const BatchTorchAsrModel& other) {
  // 1. Init the model info
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  // 2. Model copy, just copy the model ptr since:
  // PyTorch allows using multiple CPU threads during TorchScript model
  // inference, please see https://pytorch.org/docs/stable/notes/cpu_
  // threading_torchscript_inference.html
  model_ = other.model_;
  device_ = other.device_;
}

std::shared_ptr<BatchAsrModel> BatchTorchAsrModel::Copy() const {
  auto asr_model = std::make_shared<BatchTorchAsrModel>(*this);
  return asr_model;
}

void BatchTorchAsrModel::ForwardEncoder(
    const batch_feature_t& batch_feats,
    const std::vector<int>& batch_feats_lens,
    std::vector<std::vector<std::vector<float>>>& batch_topk_scores,
    std::vector<std::vector<std::vector<int32_t>>>& batch_topk_indexs) {
  // 1. Prepare libtorch required data
  int batch_size = batch_feats.size();
  int num_frames = batch_feats[0].size();
  const int feature_dim = batch_feats[0][0].size();
  torch::Tensor feats =
      torch::zeros({batch_size, num_frames, feature_dim}, torch::kFloat);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_frames; ++j) {
      torch::Tensor row =
        torch::from_blob(const_cast<float*>(batch_feats[i][j].data()),
                         {feature_dim}, torch::kFloat).clone();
      feats[i][j] = std::move(row);
    }
  }
  torch::Tensor feats_lens =
    torch::from_blob(const_cast<int*>(batch_feats_lens.data()),
                     {batch_size}, torch::kInt).clone();

  // 2. Encoder batch forward
  feats = feats.to(device_);
  feats_lens = feats_lens.to(device_);
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs = {feats, feats_lens};

  auto outputs =
      model_->get_method("batch_forward_encoder")(inputs).toTuple()->elements();
  CHECK_EQ(outputs.size(), 5);
  encoder_out_ = outputs[0].toTensor();  // (B, Tmax, dim)
  encoder_lens_ = outputs[1].toTensor();  // (B,)

  // Copy topk_scores
  auto topk_scores = outputs[3].toTensor().to(at::kCPU);
  int num_outputs = topk_scores.size(1);
  int output_dim = topk_scores.size(2);
  batch_topk_scores.resize(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    batch_topk_scores[i].resize(num_outputs);
    for (size_t j = 0; j < num_outputs; j++) {
      batch_topk_scores[i][j].resize(output_dim);
      memcpy(batch_topk_scores[i][j].data(), topk_scores[i][j].data_ptr(),
             sizeof(float) * output_dim);
    }
  }
  // copy topk_indexes
  auto topk_indexes = outputs[4].toTensor().to(at::kCPU);
  batch_topk_indexs.resize(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    batch_topk_indexs[i].resize(num_outputs);
    for (size_t j = 0; j < num_outputs; ++j) {
      batch_topk_indexs[i][j].resize(output_dim);
      memcpy(batch_topk_indexs[i][j].data(), topk_indexes[i][j].data_ptr(),
             sizeof(int) * output_dim);
    }
  }
}

void BatchTorchAsrModel::AttentionRescoring(
    const std::vector<std::vector<std::vector<int>>>& batch_hyps,
    const std::vector<std::vector<float>>& ctc_scores,
    std::vector<std::vector<float>>& attention_scores) {
  // Step 1: Prepare input for libtorch
  int batch_size = batch_hyps.size();
  int beam_size = batch_hyps[0].size();
  torch::Tensor hyps_lens_sos = torch::zeros(
      {batch_size, beam_size}, torch::kLong);
  int max_hyps_len = 0;
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < beam_size; j++) {
      int length = batch_hyps[i][j].size() + 1;
      max_hyps_len = std::max(length, max_hyps_len);
      hyps_lens_sos[i][j] = static_cast<int64_t>(length);
    }
  }

  // 1.2 add sos, eos to hyps, r_hyps
  torch::Tensor hyps_pad_sos_eos = torch::zeros(
      {batch_size, beam_size, max_hyps_len + 1}, torch::kLong);
  torch::Tensor r_hyps_pad_sos_eos = torch::zeros(
      {batch_size, beam_size, max_hyps_len + 1}, torch::kLong);
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < beam_size; j++) {
      const std::vector<int>& hyp = batch_hyps[i][j];
      hyps_pad_sos_eos[i][j][0] = sos_;
      r_hyps_pad_sos_eos[i][j][0] = sos_;
      size_t hyps_len = hyp.size();
      for (size_t k = 0; k < hyps_len; k++) {
        hyps_pad_sos_eos[i][j][k + 1] = hyp[k];
        r_hyps_pad_sos_eos[i][j][k + 1] = hyp[hyps_len - 1 - k];
      }
    }
  }

  // 1.3 ctc_scores_data
  torch::Tensor ctc_scores_tensor = torch::zeros(
      {batch_size, beam_size}, torch::kFloat);
  for (size_t i = 0; i < batch_size; ++i) {
    auto row = torch::from_blob(const_cast<float*>(ctc_scores[i].data()),
                                {beam_size}, torch::kFloat).clone();
    ctc_scores_tensor[i] = std::move(row);
  }

  // Step 2: Forward attention decoder
  hyps_pad_sos_eos = hyps_pad_sos_eos.to(device_);
  hyps_lens_sos = hyps_lens_sos.to(device_);
  r_hyps_pad_sos_eos = r_hyps_pad_sos_eos.to(device_);
  ctc_scores_tensor = ctc_scores_tensor.to(device_);
  // encoder_lens_ = encoder_lens_.to(device_);
  // encoder_out_ = encoder_out_.to(device_);
  torch::NoGradGuard no_grad;
  auto outputs = model_->run_method(
      "batch_forward_attention_decoder",
      encoder_out_, encoder_lens_,
      hyps_pad_sos_eos, hyps_lens_sos,
      r_hyps_pad_sos_eos, ctc_scores_tensor).toTuple()->elements();
  auto rescores = outputs[1].toTensor().to(at::kCPU);
#ifdef USE_GPU
  c10::cuda::CUDACachingAllocator::emptyCache();
#endif
  attention_scores.resize(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    attention_scores[i].resize(beam_size);
    memcpy(attention_scores[i].data(), rescores[i].data_ptr(),
        sizeof(float) * beam_size);
  }
}

}  // namespace wenet
