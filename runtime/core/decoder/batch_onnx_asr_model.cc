// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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


#include "decoder/batch_onnx_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <eigen3/Eigen/Eigen>

#include "utils/string.h"
#include "utils/Yaml.hpp"

namespace wenet {

Ort::Env BatchOnnxAsrModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions BatchOnnxAsrModel::session_options_ = Ort::SessionOptions();

void BatchOnnxAsrModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetInterOpNumThreads(num_threads);
}

void BatchOnnxAsrModel::GetInputOutputInfo(
    const std::shared_ptr<Ort::Session>& session,
    std::vector<const char*>* in_names, std::vector<const char*>* out_names) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session->GetInputCount();
  in_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*in_names)[i] = name;
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetOutputName(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*out_names)[i] = name;
  }
}

void BatchOnnxAsrModel::Read(const std::string& model_dir, bool is_fp16) {
  is_fp16_ = is_fp16;
  VLOG(1) << "is_fp16_ " << is_fp16_;
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  VLOG(1) << "providers.size(): " << providers.size();
  bool cuda_is_available = false;
  for (auto& prd : providers) {
    VLOG(1) << "available provider: " << prd;
    if (prd.find("CUDA") != std::string::npos) {
      cuda_is_available = true;
    }
  }
  if (!cuda_is_available) {
    VLOG(1) << "CUDA is not available! Please check your GPU settings!";
    throw std::runtime_error("CUDA is not available!");
  }
  std::string encoder_onnx_path = model_dir + "/encoder.onnx";
  std::string rescore_onnx_path = model_dir + "/decoder.onnx";
  if (is_fp16) {
    encoder_onnx_path = model_dir + "/encoder_fp16.onnx";
    rescore_onnx_path = model_dir + "/decoder_fp16.onnx";
  }

  // 1. Load sessions
  std::vector<const char*> keys{
    "device_id",
    "gpu_mem_limit",
    "arena_extend_strategy",
    "cudnn_conv_algo_search",
    "do_copy_in_default_stream",
    "cudnn_conv_use_max_workspace",
    "cudnn_conv1d_pad_to_nc1d"
  };
  std::vector<const char*> values{
    "0",
    "2147483648",
    "kSameAsRequested",
    "DEFAULT",
    "1",
    "1",
    "1"
  };
  std::cout << "prepare cuda options ...\n";
  const auto& api = Ort::GetApi();
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  OrtStatus* error = api.CreateCUDAProviderOptions(&cuda_options);
  if (error) {
    api.ReleaseStatus(error);
    throw std::runtime_error("CreateCUDAProviderOptions error");
  }
  error = api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());
  if (error) {
    api.ReleaseStatus(error);
    throw std::runtime_error("UpdateCUDAProviderOptions error");
  }
  error = api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options_, cuda_options);
  if (error) {
    api.ReleaseStatus(error);
    throw std::runtime_error("SessionOptionsAppendExecutionProvider_CUDA_V2 error");
  }
  api.ReleaseCUDAProviderOptions(cuda_options);
  std::cout << "done cuda options ...\n";

  try {
    encoder_session_ = std::make_shared<Ort::Session>(
        env_, encoder_onnx_path.c_str(), session_options_);
    rescore_session_ = std::make_shared<Ort::Session>(
        env_, rescore_onnx_path.c_str(), session_options_);
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load onnx model: " << e.what();
    exit(0);
  }
  std::cout << "read onnx model done \n";

  // 2. Read config
  std::string config_path = JoinPath(model_dir, "config.yaml");
  VLOG(1) << "Read " << config_path;
  Yaml::Node root;
  Yaml::Parse(root, config_path.c_str());
  sos_ = root["sos"].As<int>();
  eos_ = root["eos"].As<int>();
  is_bidirectional_decoder_ = root["is_bidirectional_decoder"].As<bool>();
  
  LOG(INFO) << "Onnx Model Info:";
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;

  // 3. Read model nodes
  LOG(INFO) << "Onnx Encoder:";
  GetInputOutputInfo(encoder_session_, &encoder_in_names_, &encoder_out_names_);
  LOG(INFO) << "Onnx Rescore:";
  GetInputOutputInfo(rescore_session_, &rescore_in_names_, &rescore_out_names_);
}

BatchOnnxAsrModel::BatchOnnxAsrModel(const BatchOnnxAsrModel& other) {
  // metadatas
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  is_fp16_ = other.is_fp16_;

  // sessions
  encoder_session_ = other.encoder_session_;
  rescore_session_ = other.rescore_session_;

  // node names
  encoder_in_names_ = other.encoder_in_names_;
  encoder_out_names_ = other.encoder_out_names_;
  rescore_in_names_ = other.rescore_in_names_;
  rescore_out_names_ = other.rescore_out_names_;
}

std::shared_ptr<BatchAsrModel> BatchOnnxAsrModel::Copy() const {
  auto asr_model = std::make_shared<BatchOnnxAsrModel>(*this);
  // Reset the inner states for new decoding
  return asr_model;
}

void BatchOnnxAsrModel::ForwardEncoderFunc(
    const batch_feature_t& batch_feats,
    const std::vector<int>& batch_feats_lens,
    batch_ctc_log_prob_t& out_prob) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. Prepare onnx required data
  int batch_size = batch_feats.size();
  int num_frames = batch_feats[0].size();
  int feature_dim = batch_feats[0][0].size();
  Ort::Value feats_ort{nullptr};

  // speech
  const int64_t feats_shape[3] = {batch_size, num_frames, feature_dim};
  if (is_fp16_) {
    std::vector<Ort::Float16_t> feats(batch_size * num_frames * feature_dim);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_frames; ++j) {
        for (size_t k = 0; k < feature_dim; ++k) {
          int p = i * num_frames * feature_dim + j * feature_dim + k; 
          feats[p] = Ort::Float16_t(Eigen::half(batch_feats[i][j][k]).x);
        }
      }
    }
    feats_ort = std::move(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, feats.data(), feats.size(), feats_shape, 3));
  } else {
    std::vector<float> feats;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_frames; ++j) {
        feats.insert(feats.end(), batch_feats[i][j].begin(), batch_feats[i][j].end());
      }
    }
    feats_ort = std::move(Ort::Value::CreateTensor<float>(
        memory_info, feats.data(), feats.size(), feats_shape, 3));
  }
  
  // speech_lens
  const int64_t feats_lens_shape[1] = {batch_size};
  Ort::Value feats_lens_ort = Ort::Value::CreateTensor<int>(
      memory_info, const_cast<int*>(batch_feats_lens.data()),
      batch_feats_lens.size(), feats_lens_shape, 1);

  // 2. Encoder forward
  std::vector<Ort::Value> inputs;
  for (auto name : encoder_in_names_) {
    if (!strcmp(name, "speech")) {
      inputs.push_back(std::move(feats_ort));
    } else if (!strcmp(name, "speech_lengths")) {
      inputs.push_back(std::move(feats_lens_ort));
    } 
  }

  std::vector<Ort::Value> ort_outputs = encoder_session_->Run(
      Ort::RunOptions{nullptr}, encoder_in_names_.data(), inputs.data(),
      inputs.size(), encoder_out_names_.data(), encoder_out_names_.size());

  float* ctc_log_probs = nullptr;
  auto type_info = ort_outputs[2].GetTensorTypeAndShapeInfo();
  auto out_shape = type_info.GetShape();
  int num_outputs = out_shape[1]; 
  int output_dim = out_shape[2];
  if (is_fp16_) {
    uint16_t* probs = ort_outputs[2].GetTensorMutableData<uint16_t>();
    int length = out_shape[0] * out_shape[1] * out_shape[2];
    ctc_log_probs = new float[length];
    for (size_t i = 0; i < length; ++i) {
      ctc_log_probs[i] = Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(probs[i]));
    }
  } else {
    ctc_log_probs = ort_outputs[2].GetTensorMutableData<float>();
  }

  out_prob.resize(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    out_prob[i].resize(num_outputs);
    for (size_t j = 0; j < num_outputs; j++) {
      out_prob[i][j].resize(output_dim);
      float* p = ctc_log_probs + (i * num_outputs + j) * output_dim;
      memcpy(out_prob[i][j].data(), p, sizeof(float) * output_dim);
    }
  }
  if (is_fp16_) {
    delete [] ctc_log_probs;
  }
  // 3. cache encoder outs
  encoder_outs_ = std::move(ort_outputs[0]);
}

float BatchOnnxAsrModel::ComputeAttentionScore(const float* prob,
                                          const std::vector<int>& hyp, int eos,
                                          int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void BatchOnnxAsrModel::AttentionRescoring(
    const std::vector<std::vector<std::vector<int>>>& batch_hyps,
    float reverse_weight,
    std::vector<std::vector<float>>* attention_scores) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. prepare input for onnx
  int batch_size = batch_hyps.size();
  int beam_size = batch_hyps[0].size();
  
  // 1.1 generate hyps_lens_sos data for ort
  std::vector<int> hyps_lens_sos(batch_size * beam_size, 0);  // (batch_size, beam_size)
  int max_hyps_len = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < beam_size; ++j) {
      int length = batch_hyps[i][j].size() + 1;
      max_hyps_len = std::max(length, max_hyps_len);
      hyps_lens_sos[i * beam_size + j] = length;
    }
  }

  // 1.2 generate  hyps_pad_sos
  std::vector<int64_t> hyps_pad_sos(batch_size * beam_size * max_hyps_len, 0);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < beam_size; ++j) {
      const std::vector<int>& hyps = batch_hyps[i][j];
      hyps_pad_sos[i * beam_size * max_hyps_len] = sos_;
      for (size_t k = 0; k < hyps.size(); ++k) {
        hyps_pad_sos[i * beam_size * max_hyps_len + j * max_hyps_len + k + 1] = hyps[k];
      }
    }
  }

  // 2. forward attetion decoder
  const int64_t hyps_lens_shape[] = {batch_size, beam_size};
  const int64_t hyps_pad_shape[] = {batch_size, beam_size, max_hyps_len};
  
  Ort::Value hyps_lens_tensor = Ort::Value::CreateTensor<int>(
      memory_info, hyps_lens_sos.data(), hyps_lens_sos.size(), hyps_lens_shape, 2);
  Ort::Value hyps_pad_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_pad_sos.data(), hyps_pad_sos.size(), hyps_pad_shape, 3);
  
  std::vector<Ort::Value> rescore_inputs;
  rescore_inputs.emplace_back(std::move(encoder_outs_));
  rescore_inputs.emplace_back(std::move(hyps_pad_tensor));
  rescore_inputs.emplace_back(std::move(hyps_lens_tensor));

  std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
      Ort::RunOptions{nullptr}, rescore_in_names_.data(), rescore_inputs.data(),
      rescore_inputs.size(), rescore_out_names_.data(),
      rescore_out_names_.size());

  auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> decoder_out_shape = type_info.GetShape(); //(B, beam, T2)
  float* decoder_outs_data = nullptr;
  float* r_decoder_outs_data = nullptr;
  if (is_fp16_) {
    int length = decoder_out_shape[0] * decoder_out_shape[1] * decoder_out_shape[2];
    decoder_outs_data = new float[length]();
    auto outs = rescore_outputs[0].GetTensorMutableData<uint16_t>();
    for (size_t i = 0; i < length; ++i) {
      decoder_outs_data[i] = Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(outs[i]));
    }
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      r_decoder_outs_data = new float[length]();
      auto r_outs = rescore_outputs[1].GetTensorMutableData<uint16_t>();
      for (size_t i = 0; i < length; ++i) {
        r_decoder_outs_data[i] = Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(r_outs[i]));
      }
    }
  } else {
    decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();
    }
  }
 
  int decode_out_len = decoder_out_shape[2];
  attention_scores->clear();
  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<float> Y(beam_size);
    for (size_t j = 0; j < beam_size; ++j) {
      const std::vector<int>& hyp = batch_hyps[i][j];
      float score = 0.0f;
      float* p = decoder_outs_data + (i * beam_size + j) * max_hyps_len * decode_out_len;
      score = ComputeAttentionScore(p, hyp, eos_, decode_out_len);
      float r_score = 0.0f;
      if (is_bidirectional_decoder_ && reverse_weight > 0) {
        std::vector<int> r_hyp(hyp.size());
        std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
        p = r_decoder_outs_data + (i * beam_size +j) * max_hyps_len * decode_out_len; 
        r_score = ComputeAttentionScore(p, r_hyp, eos_, decode_out_len);
      }
      Y[j] = score * (1 - reverse_weight) + r_score * reverse_weight;
    }
    attention_scores->push_back(std::move(Y));
  }
  if (is_fp16_) {
    delete [] decoder_outs_data;
    if (is_bidirectional_decoder_ && reverse_weight > 0) {
      delete [] r_decoder_outs_data;
    }
  }
}

}  // namespace wenet
