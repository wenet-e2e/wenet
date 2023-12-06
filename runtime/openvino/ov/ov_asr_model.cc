// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov/ov_asr_model.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace wenet {

static void PrintPerformanceCounts(
    std::vector<ov::ProfilingInfo> performanceData, std::ostream& stream,
    std::string deviceName, bool bshowHeader = true) {
  std::chrono::microseconds totalTime = std::chrono::microseconds::zero();
  // Print performance counts
  if (bshowHeader) {
    stream << std::endl << "performance counts:" << std::endl << std::endl;
  }
  std::ios::fmtflags fmt(std::cout.flags());
  for (const auto& it : performanceData) {
    std::string toPrint(it.node_name);
    const int maxLayerName = 30;

    if (it.node_name.length() >= maxLayerName) {
      toPrint = it.node_name.substr(0, maxLayerName - 4);
      toPrint += "...";
    }

    stream << std::setw(maxLayerName) << std::left << toPrint;
    switch (it.status) {
      case ov::ProfilingInfo::Status::EXECUTED:
        stream << std::setw(15) << std::left << "EXECUTED";
        break;
      case ov::ProfilingInfo::Status::NOT_RUN:
        stream << std::setw(15) << std::left << "NOT_RUN";
        break;
      case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
        stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
        break;
    }
    stream << std::setw(30) << std::left
           << "layerType: " + std::string(it.node_type) + " ";
    stream << std::setw(20) << std::left
           << "realTime: " + std::to_string(it.real_time.count());
    stream << std::setw(20) << std::left
           << "cpu: " + std::to_string(it.cpu_time.count());
    stream << " execType: " << it.exec_type << std::endl;
    if (it.real_time.count() > 0) {
      totalTime += it.real_time;
    }
  }
  stream << std::setw(20) << std::left
         << "Total time: " + std::to_string(totalTime.count())
         << " microseconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Full device name: " << deviceName << std::endl;
  std::cout << std::endl;
  std::cout.flags(fmt);
}

OVAsrModel::~OVAsrModel() {}

void OVAsrModel::InitEngineThreads(int core_number) {
  core_ = std::make_shared<ov::Core>();
  if (core_) {
    core_->set_property("CPU", ov::num_streams(core_number));
    core_->set_property("CPU", ov::affinity(ov::Affinity::NONE));
    core_->set_property("CPU", ov::inference_num_threads(1));
  } else {
    std::cout << "OV core not init" << std::endl;
  }
}

void OVAsrModel::Read(const std::string& model_dir) {
  std::string encoder_ir_path = model_dir + "/encoder.xml";
  std::string rescore_ir_path = model_dir + "/decoder.xml";
  std::string ctc_ir_path = model_dir + "/ctc.xml";
  auto ov_version = ov::get_openvino_version();
  std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;
  try {
    std::shared_ptr<ov::Model> encoder_model =
        core_->read_model(encoder_ir_path);
    std::map<std::string, ov::AnyMap> config;
    config["CPU"] = {};
    if (getenv("OPENVINO_PROFILE")) {
      config["CPU"].emplace(ov::enable_profiling(true));
    }
    config["CPU"].emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    // set some configurations if any
    for (auto&& item : config) {
      core_->set_property(item.first, item.second);
    }
    if (encoder_model) {
      if (encoder_model->has_rt_info("framework")) {
        auto metadata = encoder_model->get_rt_info<ov::AnyMap>("framework");

        encoder_output_size_ = metadata["output_size"].as<int>();
        num_blocks_ = metadata["num_blocks"].as<int>();
        head_ = metadata["head"].as<int>();
        cnn_module_kernel_ = metadata["cnn_module_kernel"].as<int>();
        subsampling_rate_ = metadata["subsampling_rate"].as<int>();
        right_context_ = metadata["right_context"].as<int>();
        sos_ = metadata["sos_symbol"].as<int>();
        eos_ = metadata["eos_symbol"].as<int>();
        is_bidirectional_decoder_ =
            metadata["is_bidirectional_decoder"].as<int>();
        chunk_size_ = metadata["chunk_size"].as<int>();
        num_left_chunks_ = metadata["left_chunks"].as<int>();
      } else {
        // in case that the model is exported without meta
        LOG(INFO) << "no metadata for model, use default value.";
        encoder_output_size_ = 256;
        num_blocks_ = 12;
        head_ = 4;
        cnn_module_kernel_ = 8;
        subsampling_rate_ = 4;
        right_context_ = 6;
        sos_ = 4232;
        eos_ = 4232;
        is_bidirectional_decoder_ = 1;
        chunk_size_ = 16;
        num_left_chunks_ = -1;
      }

      encoder_compile_model_ = std::make_shared<ov::CompiledModel>(
          std::move(core_->compile_model(encoder_model, "CPU")));
      // {{"PERF_COUNT", "NO"} /* YES for profile */
      // })));

      auto inputs = encoder_compile_model_->inputs();
      for (auto& input : inputs) {
        auto name = input.get_names().empty() ? "NONE" : input.get_any_name();
        encoder_inputs_map_[name] = input;
      }
      encoder_infer_ = std::make_shared<ov::InferRequest>(
          std::move(encoder_compile_model_->create_infer_request()));
    }
    std::shared_ptr<ov::Model> ctc_model = core_->read_model(ctc_ir_path);
    if (ctc_model) {
      ctc_compile_model_ = std::make_shared<ov::CompiledModel>(
          std::move(core_->compile_model(ctc_model, "CPU")));
      // {{"PERFORMANCE_HINT", "THROUGHPUT"},
      // {"PERFORMANCE_HINT_NUM_REQUESTS", 1}})));

      ctc_infer_ = std::make_shared<ov::InferRequest>(
          std::move(ctc_compile_model_->create_infer_request()));
      auto inputs = ctc_compile_model_->inputs();
      for (auto& input : inputs) {
        auto name = input.get_names().empty() ? "NONE" : input.get_any_name();
        ctc_inputs_map_[name] = input;
      }
    }
    std::shared_ptr<ov::Model> rescore_model =
        core_->read_model(rescore_ir_path);
    if (rescore_model) {
      rescore_compile_model_ = std::make_shared<ov::CompiledModel>(
          std::move(core_->compile_model(rescore_model, "CPU")));
      // {{"PERFORMANCE_HINT", "THROUGHPUT"},
      // {"PERFORMANCE_HINT_NUM_REQUESTS", 1}})));

      rescore_infer_ = std::make_shared<ov::InferRequest>(
          std::move(rescore_compile_model_->create_infer_request()));
      auto inputs = rescore_compile_model_->inputs();
      for (auto& input : inputs) {
        auto name = input.get_names().empty() ? "NONE" : input.get_any_name();
        rescore_inputs_map_[name] = input;
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load OV model" << e.what();
    exit(0);
  }

  // 2. print metadata
  LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
  LOG(INFO) << "\tnum_blocks " << num_blocks_;
  LOG(INFO) << "\thead " << head_;
  LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;
}

OVAsrModel::OVAsrModel(const OVAsrModel& other) {
  // metadatas
  // use static instead of copy
  core_ = other.core_;
  encoder_output_size_ = other.encoder_output_size_;
  num_blocks_ = other.num_blocks_;
  head_ = other.head_;
  cnn_module_kernel_ = other.cnn_module_kernel_;

  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  encoder_inputs_map_ = other.encoder_inputs_map_;
  ctc_inputs_map_ = other.ctc_inputs_map_;
  rescore_inputs_map_ = other.rescore_inputs_map_;
  encoder_compile_model_ = other.encoder_compile_model_;
  ctc_compile_model_ = other.ctc_compile_model_;
  rescore_compile_model_ = other.rescore_compile_model_;
  encoder_infer_ = std::make_shared<ov::InferRequest>(
      std::move(encoder_compile_model_->create_infer_request()));
  ctc_infer_ = std::make_shared<ov::InferRequest>(
      std::move(ctc_compile_model_->create_infer_request()));
  rescore_infer_ = std::make_shared<ov::InferRequest>(
      std::move(rescore_compile_model_->create_infer_request()));
}

std::shared_ptr<AsrModel> OVAsrModel::Copy() const {
  auto asr_model = std::make_shared<OVAsrModel>(*this);
  // Reset the inner states for new decoding
  LOG(INFO) << "Copy Once";
  asr_model->Reset();
  return asr_model;
}

void OVAsrModel::Reset() {
  offset_ = 0;
  encoder_outs_.clear();
  // Reset att_cache
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;
    offset_ = required_cache_size;
    att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                          encoder_output_size_ / head_ * 2,
                      0.0);
    ov::Shape att_cache_shape = {num_blocks_, head_, required_cache_size,
                                 encoder_output_size_ / head_ * 2};
    att_cache_ov_ =
        ov::Tensor(ov::element::f32, att_cache_shape, att_cache_.data());
  } else {
    // resize vector to 0 will introcue nullptr under some compiler
    // optimizations if you pass nullptr to ov Tensor, it will try to reuse a
    // nullptr To avoid such problem, just let OV Tensor allocate this memory.
    att_cache_.resize(1, 0.0);
    ov::Shape att_cache_shape = {num_blocks_, head_, 0,
                                 encoder_output_size_ / head_ * 2};
    att_cache_ov_ =
        ov::Tensor(ov::element::f32, att_cache_shape, att_cache_.data());
  }

  // Reset cnn_cache
  cnn_cache_.resize(
      num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
  ov::Shape cnn_cache_shape = {num_blocks_, 1, encoder_output_size_,
                               cnn_module_kernel_ - 1};
  cnn_cache_ov_ =
      ov::Tensor(ov::element::f32, cnn_cache_shape, cnn_cache_.data());
}

void OVAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  // 1. Prepare OV required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  std::vector<float> feats;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    for (size_t j = 0; j < feature_dim; ++j) {
      feats.emplace_back(cached_feature_[i][j]);
    }
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    for (size_t j = 0; j < feature_dim; ++j) {
      feats.emplace_back(chunk_feats[i][j]);
    }
  }

  ov::Shape feats_shape = {1, num_frames, feature_dim};
  ov::Tensor feats_ov = ov::Tensor(ov::element::f32, feats_shape, feats.data());

  // offset
  int64_t offset_int64 = static_cast<int64_t>(offset_);
  ov::Tensor offset_ov =
      ov::Tensor(ov::element::i64, ov::Shape(), &offset_int64);

  // required_cache_size
  int64_t required_cache_size = chunk_size_ * num_left_chunks_;
  ov::Tensor required_cache_size_ov =
      ov::Tensor(ov::element::i64, ov::Shape(), &required_cache_size);

  // att_mask
  ov::Tensor att_mask_ov;
  std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
    ov::Shape att_mask_shape = {1, 1, required_cache_size + chunk_size_};
    att_mask_ov = ov::Tensor(ov::element::boolean, att_mask_shape,
                             reinterpret_cast<bool*>(att_mask.data()));
  }

  // set input tensor
  size_t idx = 0;
  std::map<std::string, ov::Output<const ov::Node>>::iterator it =
      encoder_inputs_map_.begin();
  while (it != encoder_inputs_map_.end()) {
    if (it->first == "chunk") {
      encoder_infer_->set_tensor(it->second, feats_ov);
    } else if (it->first == "offset") {
      encoder_infer_->set_tensor(it->second, offset_ov);
    } else if (it->first == "required_cache_size") {
      encoder_infer_->set_tensor(it->second, required_cache_size_ov);
    } else if (it->first == "att_cache") {
      encoder_infer_->set_tensor(it->second, att_cache_ov_);
    } else if (it->first == "cnn_cache") {
      encoder_infer_->set_tensor(it->second, cnn_cache_ov_);
    } else if (it->first == "att_mask") {
      encoder_infer_->set_tensor(it->second, att_mask_ov);
    }
    ++it;
  }
  try {
    encoder_infer_->infer();
  } catch (std::exception& ex) {
    std::cout << ex.what() << std::endl;
  }

  const ov::Tensor ov_output = encoder_infer_->get_output_tensor(0);
  offset_ += static_cast<int>(ov_output.get_shape()[1]);
  att_cache_ov_ = encoder_infer_->get_output_tensor(1);
  cnn_cache_ov_ = encoder_infer_->get_output_tensor(2);
  encoder_outs_.push_back(ov_output);
  // Please use set_input_tensor if you are set by index not by string.

  // ctc_infer_->set_input_tensor(0, ov_output);
  ctc_infer_->set_tensor(ctc_inputs_map_.begin()->first, ov_output);
  ctc_infer_->infer();

  const ov::Tensor ctc_output = ctc_infer_->get_output_tensor();
  float* logp_data = static_cast<float*>(ctc_output.data());
  int num_outputs = ctc_output.get_shape()[1];
  int output_dim = ctc_output.get_shape()[2];

  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
           sizeof(float) * output_dim);
  }
}

float OVAsrModel::ComputeAttentionScore(const float* prob,
                                        const std::vector<int>& hyp, int eos,
                                        int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void OVAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
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
    float* encoder_outs_data = static_cast<float*>(encoder_outs_[i].data());
    ov::Shape shape_info = encoder_outs_[i].get_shape();
    for (int j = 0; j < encoder_outs_[i].get_size(); j++) {
      rescore_input.emplace_back(encoder_outs_data[j]);
    }
    encoder_len += shape_info[1];
  }

  ov::Shape decode_input_shape = {1, encoder_len, encoder_output_size_};

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

  ov::Tensor decoder_input_tensor =
      ov::Tensor(ov::element::f32, decode_input_shape, rescore_input.data());
  ov::Shape hyps_pad_shape = {num_hyps, max_hyps_len};
  ov::Tensor hyps_pad_tensor =
      ov::Tensor(ov::element::i64, hyps_pad_shape, hyps_pad.data());
  ov::Shape hyps_lens_shape = {num_hyps};
  ov::Tensor hyps_lens_tensor =
      ov::Tensor(ov::element::i64, hyps_lens_shape, hyps_lens.data());

  rescore_infer_->set_tensor("hyps", hyps_pad_tensor);
  rescore_infer_->set_tensor("hyps_lens", hyps_lens_tensor);
  rescore_infer_->set_tensor("encoder_out", decoder_input_tensor);

  rescore_infer_->infer();

  const ov::Tensor decoder_score = rescore_infer_->get_output_tensor(0);
  const ov::Tensor r_decoder_score = rescore_infer_->get_output_tensor(1);
  float* decoder_outs_data = static_cast<float*>(decoder_score.data());
  float* r_decoder_outs_data = static_cast<float*>(r_decoder_score.data());

  int decode_out_len = decoder_score.get_shape()[2];

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
