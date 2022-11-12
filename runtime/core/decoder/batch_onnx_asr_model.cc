// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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


#include "decoder/batch_onnx_asr_model.h"

#include <immintrin.h>
#include <algorithm>
#include <memory>
#include <utility>

#include "glog/logging.h"
#include "utils/string.h"
#include "utils/Yaml.hpp"
#include "utils/timer.h"

namespace wenet {

Ort::Env BatchOnnxAsrModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "");
Ort::SessionOptions BatchOnnxAsrModel::session_options_ = Ort::SessionOptions();
Ort::RunOptions BatchOnnxAsrModel::run_option_ = Ort::RunOptions();
std::vector<Ort::AllocatedStringPtr> BatchOnnxAsrModel::node_names_;

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
    auto name = session->GetInputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << name.get()
              << " type=" << type << " dims=" << shape.str();
    node_names_.push_back(std::move(name));
    (*in_names)[i] = node_names_.back().get();
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto name = session->GetOutputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << name.get()
              << " type=" << type << " dims=" << shape.str();
    node_names_.push_back(std::move(name));
    (*out_names)[i] = node_names_.back().get();
  }
}

void BatchOnnxAsrModel::Read(const std::string& model_dir,
    bool is_fp16, int gpu_id) {
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

  // release GPU memory:
  // https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
  // 1. Not allocate weights memory through the arena
  session_options_.AddConfigEntry(
      kOrtSessionOptionsUseDeviceAllocatorForInitializers, "1");
  // 2. Configure the arena to have high enough initial chunk
  // to support most Run() calls. See "initial_chunk_size_bytes"
  const char* keys[] = {
    "max_mem", "arena_extend_strategy", "initial_chunk_size_bytes",
    "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
  const size_t values[] = {0, 0, 1024, 0, 256};

  OrtArenaCfg* arena_cfg = nullptr;
  const auto& api = Ort::GetApi();
  auto zz = api.CreateArenaCfgV2(keys, values, 5, &arena_cfg);
  std::unique_ptr<OrtArenaCfg, decltype(api.ReleaseArenaCfg)> rel_arena_cfg(
      arena_cfg, api.ReleaseArenaCfg);

  OrtCUDAProviderOptions cuda_options{};

  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search =
    OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
  // cuda_options.gpu_mem_limit = 16 * 1024 * 1024 * 1024ul;
  cuda_options.arena_extend_strategy = 1;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = 0;
  cuda_options.user_compute_stream = nullptr;
  // TODO(veelion): arena_cfg didn't work, it blocked when session.Run()
  // Just comment this out until find a work way.
  // cuda_options.default_memory_arena_cfg = arena_cfg;
  session_options_.AppendExecutionProvider_CUDA(cuda_options);

  /* TODO(veelion): use OrtCUDAProviderOptionsV2 until it support ArenaCfg
  // 1. Load sessions
  // config for CUDA
  std::string device_id = std::to_string(gpu_id);
  std::vector<const char*> keys2{
    "device_id",
    "gpu_mem_limit",
    "arena_extend_strategy",
    "cudnn_conv_algo_search",
    "do_copy_in_default_stream",
    "cudnn_conv_use_max_workspace",
    "cudnn_conv1d_pad_to_nc1d"  // supported from 1.12.0
  };
  std::vector<const char*> values2{
    device_id.data(),
    //"2147483648",
    "8589934592",
    "kSameAsRequested",
    "DEFAULT",
    "1",
    "1",
    "1"
  };

  const auto& api = Ort::GetApi();
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
    cuda_options, keys2.data(), values2.data(), keys2.size()));
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
    session_options_, cuda_options));
  api.ReleaseCUDAProviderOptions(cuda_options);
  */

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

void BatchOnnxAsrModel::ForwardEncoder(
    const batch_feature_t& batch_feats,
    const std::vector<int>& batch_feats_lens,
    std::vector<std::vector<std::vector<float>>>* batch_topk_scores,
    std::vector<std::vector<std::vector<int32_t>>>* batch_topk_indexs) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. Prepare onnx required data
  int batch_size = batch_feats.size();
  int num_frames = batch_feats[0].size();
  int feature_dim = batch_feats[0][0].size();

  // generate data for CreateTensor
  Ort::Value feats_ort{nullptr};
  // https://github.com/microsoft/onnxruntime/issues/9629#issuecomment-963828881
  // Ort::Value::CreateTensor does NOT copy the data
  std::vector<Ort::Float16_t> feats_fp16;  // for holding feats of fp16
  std::vector<float> feats_fp32;  // for holding feats of float

  // speech
  const int64_t feats_shape[3] = {batch_size, num_frames, feature_dim};
  Timer timer;
  if (is_fp16_) {
    feats_fp16.resize(batch_size * num_frames * feature_dim);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_frames; ++j) {
        for (size_t k = 0; k < feature_dim; ++k) {
          int p = i * num_frames * feature_dim + j * feature_dim + k;
          feats_fp16[p] = Ort::Float16_t(_cvtss_sh(batch_feats[i][j][k], 0));
        }
      }
    }
    auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info,
        feats_fp16.data(),
        feats_fp16.size(),
        feats_shape, 3);
    feats_ort = std::move(tensor);
    VLOG(1) << "feats to fp16 takes " << timer.Elapsed() << " ms.";
  } else {
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_frames; ++j) {
        feats_fp32.insert(feats_fp32.end(), batch_feats[i][j].begin(),
            batch_feats[i][j].end());
      }
    }
    feats_ort = std::move(Ort::Value::CreateTensor<float>(
        memory_info, feats_fp32.data(), feats_fp32.size(), feats_shape, 3));
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

  timer.Reset();
  // Ort::RunOptions ro;
  // ro.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
  std::vector<Ort::Value> ort_outputs = encoder_session_->Run(
      Ort::RunOptions{nullptr}, encoder_in_names_.data(), inputs.data(),
      inputs.size(), encoder_out_names_.data(), encoder_out_names_.size());
  VLOG(1) << "\tencoder ->Run() takes " << timer.Elapsed() << " ms.";

  // get topk_scores
  auto out_shape = ort_outputs[3].GetTensorTypeAndShapeInfo().GetShape();
  int num_outputs = out_shape[1];
  int output_dim = out_shape[2];
  float* topk_scores_ptr = nullptr;
  std::vector<float> topk_scores_data;  // for holding topk_scores in fp16
  if (is_fp16_) {
    timer.Reset();
    auto probs = ort_outputs[3].GetTensorMutableData<uint16_t>();
    int length = out_shape[0] * out_shape[1] * out_shape[2];
    topk_scores_data.resize(length);
    for (size_t i = 0; i < length; ++i) {
      topk_scores_data[i] = _cvtsh_ss(probs[i]);
    }
    topk_scores_ptr = topk_scores_data.data();
    VLOG(1) << "topk_scores from GPU-fp16 to float takes " << timer.Elapsed()
      << " ms. data lenght " << length;
  } else {
    topk_scores_ptr = ort_outputs[3].GetTensorMutableData<float>();
  }

  batch_topk_scores->resize(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    (*batch_topk_scores)[i].resize(num_outputs);
    for (size_t j = 0; j < num_outputs; j++) {
      (*batch_topk_scores)[i][j].resize(output_dim);
      float* p = topk_scores_ptr + (i * num_outputs + j) * output_dim;
      memcpy((*batch_topk_scores)[i][j].data(), p, sizeof(float) * output_dim);
    }
  }
  // get batch_topk_indexs
  std::vector<int32_t> topk_indexs_data;  // for holding topk_indexs from fp16
  timer.Reset();
  auto probs = ort_outputs[4].GetTensorMutableData<int64_t>();
  int length = out_shape[0] * out_shape[1] * out_shape[2];
  topk_indexs_data.resize(length);
  for (size_t i = 0; i < length; ++i) {
    topk_indexs_data[i] = probs[i];
  }
  int32_t* topk_indexs_ptr = topk_indexs_data.data();
  VLOG(1) << "topk_indexs from GPU-fp16 to float takes "
          << timer.Elapsed() << " ms. data lenght " << length;

  batch_topk_indexs->resize(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    (*batch_topk_indexs)[i].resize(num_outputs);
    for (size_t j = 0; j < num_outputs; j++) {
      (*batch_topk_indexs)[i][j].resize(output_dim);
      int32_t* p = topk_indexs_ptr + (i * num_outputs + j) * output_dim;
      memcpy((*batch_topk_indexs)[i][j].data(), p,
          sizeof(int32_t) * output_dim);
    }
  }
  // 3. cache encoder outs
  encoder_outs_ = std::move(ort_outputs[0]);
  encoder_outs_lens_ = std::move(ort_outputs[1]);
}

void BatchOnnxAsrModel::AttentionRescoring(
    const std::vector<std::vector<std::vector<int>>>& batch_hyps,
    const std::vector<std::vector<float>>& ctc_scores,
    std::vector<std::vector<float>>* attention_scores) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. prepare input for onnx
  int batch_size = batch_hyps.size();
  int beam_size = batch_hyps[0].size();

  // 1.1 generate hyps_lens_sos data for ort  (batch_size, beam_size)
  std::vector<int> hyps_lens_sos(batch_size * beam_size, 0);
  int max_hyps_len = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < beam_size; ++j) {
      int length = batch_hyps[i][j].size() + 1;
      max_hyps_len = std::max(length, max_hyps_len);
      hyps_lens_sos[i * beam_size + j] = length;
    }
  }

  // 1.2 generate  hyps_pad_sos_eos, r_hyps_pad_sos_eos
  std::vector<int64_t> hyps_pad_sos_eos(
      batch_size * beam_size * (max_hyps_len + 1), 0);
  std::vector<int64_t> r_hyps_pad_sos_eos(
      batch_size * beam_size * (max_hyps_len + 1), 0);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < beam_size; ++j) {
      const std::vector<int>& hyps = batch_hyps[i][j];
      hyps_pad_sos_eos[i * beam_size * max_hyps_len] = sos_;
      size_t hyps_len = hyps.size();
      for (size_t k = 0; k < hyps_len; ++k) {
        size_t p = i * beam_size * max_hyps_len + j * max_hyps_len + k + 1;
        hyps_pad_sos_eos[p] = hyps[k];
        r_hyps_pad_sos_eos[p] = hyps[hyps_len - 1 - k];
      }
      size_t p = i * beam_size * max_hyps_len +
        j * max_hyps_len + hyps.size() + 1;
      hyps_pad_sos_eos[p] = eos_;
      r_hyps_pad_sos_eos[p] = eos_;
    }
  }

  // 1.3 ctc_scores_data
  Ort::Value ctc_scores_tensor{nullptr};
  std::vector<Ort::Float16_t> ctc_fp16;
  std::vector<float> ctc_fp32;
  const int64_t ctc_shape[] = {batch_size, beam_size};
  if (is_fp16_) {
    ctc_fp16.resize(batch_size * beam_size);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < beam_size; ++j) {
        int p = i * beam_size + j;
        ctc_fp16[p] = Ort::Float16_t(_cvtss_sh(ctc_scores[i][j], 0));
      }
    }
    ctc_scores_tensor = std::move(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, ctc_fp16.data(), ctc_fp16.size(), ctc_shape, 2));
  } else {
    ctc_fp32.resize(batch_size * beam_size);
    for (size_t i = 0; i < batch_size; ++i) {
      memcpy(ctc_fp32.data() + i * beam_size,
          ctc_scores[i].data(), sizeof(float) * beam_size);
    }
    ctc_scores_tensor = std::move(Ort::Value::CreateTensor<float>(
        memory_info, ctc_fp32.data(), ctc_fp32.size(), ctc_shape, 2));
  }

  // 2. forward attetion decoder
  const int64_t hyps_lens_shape[] = {batch_size, beam_size};
  const int64_t hyps_pad_shape[] = {batch_size, beam_size, max_hyps_len};

  Ort::Value hyps_lens_tensor = Ort::Value::CreateTensor<int>(
      memory_info, hyps_lens_sos.data(),
      hyps_lens_sos.size(), hyps_lens_shape, 2);
  Ort::Value hyps_pad_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, hyps_pad_sos_eos.data(),
      hyps_pad_sos_eos.size(), hyps_pad_shape, 3);
  Ort::Value r_hyps_pad_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, r_hyps_pad_sos_eos.data(),
      r_hyps_pad_sos_eos.size(), hyps_pad_shape, 3);

  std::vector<Ort::Value> rescore_inputs;
  for (auto name : rescore_in_names_) {
    if (!strcmp(name, "encoder_out")) {
      rescore_inputs.push_back(std::move(encoder_outs_));
    } else if (!strcmp(name, "encoder_out_lens")) {
      rescore_inputs.push_back(std::move(encoder_outs_lens_));
    } else if (!strcmp(name, "hyps_pad_sos_eos")) {
      rescore_inputs.push_back(std::move(hyps_pad_tensor));
    } else if (!strcmp(name, "hyps_lens_sos")) {
      rescore_inputs.push_back(std::move(hyps_lens_tensor));
    } else if (!strcmp(name, "r_hyps_pad_sos_eos")) {
      rescore_inputs.push_back(std::move(r_hyps_pad_tensor));
    } else if (!strcmp(name, "ctc_score")) {
      rescore_inputs.push_back(std::move(ctc_scores_tensor));
    } else {
      VLOG(1) << "invalid input name " << name;
    }
  }

  Timer timer;
  Ort::RunOptions ro;
  ro.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
  std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
      ro, rescore_in_names_.data(), rescore_inputs.data(),
      rescore_inputs.size(), rescore_out_names_.data(),
      rescore_out_names_.size());
  VLOG(1) << "decoder->Run() takes " << timer.Elapsed() << " ms.";

  // (B, beam, T2)
  auto scores_shape = rescore_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  attention_scores->resize(scores_shape[0]);
  if (is_fp16_) {
    Timer timer;
    int length = scores_shape[0] * scores_shape[1];
    auto outs = rescore_outputs[1].GetTensorMutableData<Ort::Float16_t>();
    for (size_t i = 0; i < scores_shape[0]; ++i) {
      (*attention_scores)[i].resize(scores_shape[1]);
      for (size_t j = 0; j < scores_shape[1]; ++j) {
        (*attention_scores)[i][j] = _cvtsh_ss(
            outs[i * scores_shape[1] + j].value);
      }
    }
    VLOG(1) << "decoder_out from fp16 to float takes "
            << timer.Elapsed() << " ms. data length " << length;
  } else {
    auto outs = rescore_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i < scores_shape[0]; ++i) {
      (*attention_scores)[i].resize(scores_shape[1]);
      memcpy((*attention_scores)[i].data(), outs + i * scores_shape[1],
          sizeof(float) * scores_shape[1]);
    }
  }
}

}  // namespace wenet
