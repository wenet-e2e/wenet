// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Han Qi (qihan@baidu.com, Kunlunxin Inc)
//                    Hehe Pan (panhehe@baidu.com, Kunlunxin Inc)
//                    Zikui Yan (yanzikui@baidu.com, Kunlunxin Inc)
// All Rights Reserved.
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

#include "xpu_asr_model.h"  // NOLINT

#include <algorithm>
#include <fstream>
#include <memory>
#include <utility>

#include "utils/string.h"

namespace wenet {

void XPUAsrModel::SetEngineThreads(int num_threads) {
  real_threads_number = num_threads;
}

void XPUAsrModel::SetDeviceId(int dev_id) { device_id_ = dev_id; }

void XPUAsrModel::Read(const std::string& model_dir) {
  // init xpu runtime params
  ctx_xpu_ptr = std::make_shared<api::Context>(api::kXPU2);
  RAII_GUARD.reset(new api::ctx_guard(ctx_xpu_ptr.get()));

  // For XPU, model_dir is params_dir, which is used to store weights for every
  // layer.
  std::string weight_dir = model_dir + "/model_weights/";
  std::string weight_info_txt_path = weight_dir + "/weights_info.txt";

  LOG(INFO) << "\e[1;34mXPU weight_dir is: " << weight_dir << "\e[0m\n";
  if (!std::ifstream(weight_info_txt_path.c_str()).good()) {
    LOG(FATAL) << "weight_info_txt: " << weight_info_txt_path
               << " NOT exist !!!\n";
  }

  // 1. Load weight for every layer
  init_encoder_params<T, TW>(weight_dir, encoder_param);
  init_decoder_params<T, TW>(weight_dir, decoder_param);

  // 2. Read metadata
  // TODO(panhehe): Load following parameters from config file or
  // encoder/decoder params.
  subsampling_rate_ = 4;
  right_context_ = 6;
  sos_ = 5538;
  eos_ = 5538;
  is_bidirectional_decoder_ = 1;

  LOG(INFO) << "======= XPU Kunlun Model Info: =======";
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
}

XPUAsrModel::XPUAsrModel(const XPUAsrModel& other) {
  // 1. Init the model info
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  l3ptr = other.l3ptr;
  real_threads_number = other.real_threads_number;
  device_id_ = other.device_id_;
  ctx_xpu_ptr = other.ctx_xpu_ptr;
  RAII_GUARD = other.RAII_GUARD;
  encoder_param = other.encoder_param;
  decoder_param = other.decoder_param;
  stream = other.stream;
  // other member variables may not need to copy here
}

std::shared_ptr<AsrModel> XPUAsrModel::Copy() const {
  auto asr_model = std::make_shared<XPUAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

void XPUAsrModel::Reset() {
  offset_ = 0;
  encoder_out = nullptr;
  ctc_probs = nullptr;
  cached_feature_.clear();
  // Reset att_cache
  att_cache_.resize(0, 0.0);
  cnn_cache_.resize(0, 0.0);
}

void XPUAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  // Set Device Id
  LOG(INFO) << "Now Use XPU:" << device_id_ << "!\n";
  xpu_set_device(device_id_);

  // 1. Prepare XPU required data, splice cached_feature_ and chunk_feats
  // The first dimension is for batchsize, which is 1.
  // chunk

  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();

  std::vector<int> feats_length_shape = {1};
  std::vector<int> feats_length_data = {num_frames};
  input_lenghts_cpu_info =
      std::make_tuple(feats_length_data, feats_length_shape);

  std::vector<int> feats_data_shape = {1, num_frames, feature_dim};
  std::vector<float> feats_data_cpu;
  feats_data_cpu.reserve(1 * num_frames * feature_dim);
  // convert 2d-vector to 1d-vector
  for (auto& row : chunk_feats) {
    auto end_iter = feats_data_cpu.end();
    feats_data_cpu.insert(end_iter, row.cbegin(), row.cend());
  }

  float* input_xpu_data = get_xpu_data<float>("wav_test", feats_data_cpu);
  input_xpu_info = std::make_tuple(input_xpu_data, feats_data_shape);

  // init L3 Memory
  int ret = 0;
  real_threads_number = 1;
  int nsdnn = real_threads_number > 1 ? 2 : 6;
  int ncluster = real_threads_number > 1 ? 2 : 8;
  for (int i = 0; i < real_threads_number; i++) {
    ret = xpu_stream_create(&stream);
    ctx_xpu_ptr->xpu_stream = stream;
    ctx_xpu_ptr->set_nsdnn(nsdnn);
    ctx_xpu_ptr->set_ncluster(ncluster);
  }

  std::shared_ptr<api::Context> ctx_xpu = ctx_xpu_ptr;

  // get input speech info and data
  batch = feats_data_shape.at(0);  // batch = 1
  max_seqlen = feats_data_shape.at(1);

  xpu_mask_info_float = create_mask_according_speech_length<float>(
      feats_length_data, max_seqlen, ctx_xpu->xpu_stream);

  ret = xpu_wait(ctx_xpu->xpu_stream);
  CHECK_RET(ret);

  q_seqlen = ((max_seqlen - 1) / 2 - 1) / 2;

  // Encoder run
  int att_dim = encoder_param.head_num * encoder_param.head_dim;
  int ctc_dim = encoder_param.ctc_dim;

  LOG(INFO) << "\t max_seqlen is " << max_seqlen << "\n";
  LOG(INFO) << "\t q_seqlen   is " << q_seqlen << "\n";
  LOG(INFO) << "\t att_dim    is " << att_dim << "\n";
  LOG(INFO) << "\t ctc_dim    is " << ctc_dim << "\n";

  // T is float16
  encoder_out = RAII_GUARD->alloc<T>(batch * q_seqlen * att_dim);
  ctc_probs = RAII_GUARD->alloc<T>(batch * q_seqlen * ctc_dim);

  // 2. Encoder chunk forward, including ctc_activation
  // get encoder_out & ctc_probs
  ret = xpu::wenet::conformer_encoder_wenet<T, TW, int16_t>(
      ctx_xpu.get(), input_xpu_data, feats_data_shape, encoder_out, ctc_probs,
      encoder_param, xpu_mask_info_float);
  CHECK_RET(ret);

  // Copy to output(cpu)
  int num_outputs = q_seqlen;
  int output_dim = ctc_dim;
  out_prob->resize(num_outputs);

  float* logp = RAII_GUARD->alloc<float>(batch * q_seqlen * ctc_dim);
  // cast T to float32
  ret = api::cast_v2<T, float>(ctx_xpu.get(), ctc_probs, logp,
                               batch * q_seqlen * ctc_dim);
  CHECK_RET(ret);
  ret = xpu_wait(ctx_xpu->xpu_stream);
  CHECK_RET(ret);

  // xpu_memcpy logp from device to host
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    ret = xpu_memcpy(reinterpret_cast<void*>((*out_prob)[i].data()),
                     logp + output_dim * i, output_dim * sizeof(float),
                     XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    CHECK_RET(ret);
  }
}

float XPUAsrModel::ComputeAttentionScore(const float* prob,
                                         const std::vector<int>& hyp, int eos,
                                         int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void XPUAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                     float reverse_weight,
                                     std::vector<float>* rescoring_score) {
  CHECK(rescoring_score != nullptr);
  int num_hyps = hyps.size();
  rescoring_score->resize(num_hyps, 0.0f);

  if (num_hyps == 0) {
    return;
  }

  if (encoder_out == nullptr) {
    return;
  }

  int beam_size = encoder_param.beam_size;
  int new_bs = batch * beam_size;

  std::vector<int64_t> hyps_lens;
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_lens.emplace_back(static_cast<int64_t>(length));
  }
  LOG(INFO) << "\t num_hyps  is " << num_hyps << "\n";
  LOG(INFO) << "\t beam_size is " << beam_size << "\n";
  LOG(INFO) << "\t new_bs    is " << new_bs << "\n";
  LOG(INFO) << "\t max_hyps_len is " << max_hyps_len << "\n";

  // pad hyps
  std::vector<int> hyps_pad_cpu(max_hyps_len * beam_size);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_pad_cpu.emplace_back(sos_);
    size_t j = 0;
    for (; j < hyp.size(); ++j) {
      hyps_pad_cpu.emplace_back(hyp[j]);
    }
    if (j == max_hyps_len - 1) {
      continue;
    }
    for (; j < max_hyps_len - 1; ++j) {
      hyps_pad_cpu.emplace_back(0);
    }
  }
  int* hyps_xpu = RAII_GUARD->alloc<int>(new_bs * q_seqlen);
  int max_target_len = max_hyps_len;
  // xpu_memcpy hyps_pad_cup to device
  int ret = xpu_memcpy(hyps_xpu, reinterpret_cast<void*>(hyps_pad_cpu.data()),
                       max_target_len * new_bs * sizeof(int),
                       XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  CHECK_RET(ret);

  // Decoder
  int att_dim = encoder_param.head_num * encoder_param.head_dim;
  int ctc_dim = encoder_param.ctc_dim;
  int pad_target_len = decoder_param.add_sos_num + max_target_len;
  float* character_scores =
      RAII_GUARD->alloc<float>(new_bs * pad_target_len * ctc_dim);
  ret = xpu::wenet::conformer_decoder_wenet<T, TW, int16_t>(
      ctx_xpu_ptr.get(), encoder_out, {batch, q_seqlen, att_dim},
      std::get<0>(xpu_mask_info_float), hyps_xpu, {new_bs, max_target_len},
      character_scores, decoder_param);
  CHECK_RET(ret);
  ret = xpu_wait(ctx_xpu_ptr->xpu_stream);
  CHECK_RET(ret);

  // xpu_memcpy from xpu device to host
  std::vector<float> decoder_out(new_bs * pad_target_len * ctc_dim);
  ret = xpu_memcpy(&decoder_out[0], character_scores,
                   new_bs * max_target_len * ctc_dim * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  CHECK_RET(ret);
  ret = xpu_wait(ctx_xpu_ptr->xpu_stream);
  CHECK_RET(ret);

  // cal score
  float* decoder_outs_data = decoder_out.data();
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left to right decoder score
    // ctc_dim maybe equal to decode_out_len
    score = ComputeAttentionScore(
        decoder_outs_data + max_target_len * ctc_dim * i, hyp, eos_, ctc_dim);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    // reverse_weight is 0 ; so the codes in if-condition is be ignored.
    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =
        score * (1 - reverse_weight) + r_score * reverse_weight;
  }
}

}  // namespace wenet
