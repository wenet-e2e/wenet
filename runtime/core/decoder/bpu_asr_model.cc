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


#include "decoder/bpu_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "utils/string.h"

namespace wenet {

void BpuAsrModel::GetInputOutputInfo(
    const std::vector<std::shared_ptr<DNNTensor>>& input,
    const std::vector<std::shared_ptr<DNNTensor>>& output) {
  // Input info
  for (size_t i = 0; i < input.size(); ++i) {
    auto& shapes = input[i]->properties.validShape.dimensionSize;
    std::string layout = (input[i]->properties.tensorLayout ==
        hbDNNTensorLayout::HB_DNN_LAYOUT_NHWC ? "NHWC" : "NCHW");
    LOG(INFO) << "\tInput-" << i << ": Shape [" << shapes[0] << ","
              << shapes[1] << "," << shapes[2] << "," << shapes[3]
              << "], Layout [" << layout << "]";
  }
  // Output info
  for (size_t i = 0; i < output.size(); ++i) {
    auto& shapes = output[i]->properties.validShape.dimensionSize;
    std::string layout = (output[i]->properties.tensorLayout ==
        hbDNNTensorLayout::HB_DNN_LAYOUT_NHWC ? "NHWC" : "NCHW");
    LOG(INFO) << "\tOutput-" << i << ": Shape [" << shapes[0] << ","
              << shapes[1] << "," << shapes[2] << "," << shapes[3]
              << "], Layout [" << layout << "]";
  }
}

void BpuAsrModel::Read(const std::string& model_dir) {
  std::string encoder_model_path = model_dir + "/encoder.bin";
  std::string ctc_model_path = model_dir + "/ctc.bin";

  // 1. Load models
  ModelManager* model_manager = ModelManager::GetInstance();
  std::vector<Model*> models;
  model_manager->Load(models, encoder_model_path);
  encoder_model_ = model_manager_->GetModel([](Model* model) {
    return model->GetName().find("encoder") != std::string::npos;
  });
  model_manager->Load(models, ctc_model_path);
  ctc_model_ = model_manager_->GetModel([](Model* model) {
    return model->GetName().find("ctc") != std::string::npos;
  });

  // 2. Init input/output tensors
  AllocMemory(&encoder_input_, &encoder_output_, encoder_model_);
  AllocMemory(&ctc_input_, &ctc_output_, ctc_model_);
  Reset();

  // 3. Read model input/output nodes
  LOG(INFO) << "BPU Encoder:";
  GetInputOutputInfo(encoder_input_, encoder_output_);
  LOG(INFO) << "BPU CTC:";
  GetInputOutputInfo(ctc_input_, ctc_output_);

  // 4. Parse metadatas
  right_context_ = 14;    // NOTE(xcsong): Only support 1/8 subsample, since
  subsampling_rate_ = 8;  //   1/4 subsample is too slow on edge-devices.
  sos_ = ctc_output_[0]->properties.validShape.dimensionSize[1] - 1;
  eos_ = sos_;
  chunk_size_ = ctc_input_[0]->properties.validShape.dimensionSize[3];
  num_left_chunks_ = encoder_input_[2]->properties.validShape.dimensionSize[3];
      / chunk_size_;
  hidden_dim_ = ctc_input_[0]->properties.validShape.dimensionSize[1];
  int frames = (chunk_size_ - 1) * subsampling_rate_ + right_context_ + 1;
  CHECK_EQ(frames, encoder_input_[0]->properties.validShape.dimensionSize[2]) <<
      "NOTE(xcsong): Only support 1/8 subsample, since 1/4 subsample" <<
      " is too slow on edge-devices.";
  LOG(INFO) << "Bpu Model Info:";
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tis bidirectional decoder " << is_bidirectional_decoder_;
  LOG(INFO) << "\thidden_dim " << hidden_dim_;
}

BpuAsrModel::BpuAsrModel(const BpuAsrModel& other) {
  // metadatas (BaseClass)
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  // metadatas (ChileClass)
  hidden_dim_ = other.hidden_dim_;
  chunk_id_ = other.chunk_id_;

  // models, NOTE(xcsong): in/out tensors & managers are not copied here.
  encoder_model_ = other.encoder_model_;
  ctc_model_ = other.ctc_model_;
}

std::shared_ptr<AsrModel> BpuAsrModel::Copy() const {
  auto asr_model = std::make_shared<BpuAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->AllocMemory(&(asr_model->encoder_input_),
      &(asr_model->encoder_output_), encoder_model_);
  asr_model->AllocMemory(&(asr_model->ctc_input_),
      &(asr_model->ctc_output_), ctc_model_);
  asr_model->Reset();
  return asr_model;
}

void BpuAsrModel::AllocMemory(
    std::vector<std::shared_ptr<DNNTensor>>* inputs,
    std::vector<std::shared_ptr<DNNTensor>>* outputs,
    Model* model) {
  size_t input_counts = model->GetInputCount();
  inputs->resize(input_counts);
  for (size_t i = 0; i < input_counts; i++) {
    inputs->at(i).reset(new DNNTensor);
    auto& item = inputs->at(i);
    model->GetInputTensorProperties(item->properties, i);
    hbSysAllocCachedMem(&(item->sysMem[0]), item->properties.alignedByteSize);
  }
  size_t output_counts = model->GetOutputCount();
  outputs->resize(output_counts);
  for (size_t i = 0; i < output_counts; i++) {
    outputs->at(i).reset(new DNNTensor);
    auto& item = outputs->at(i);
    model->GetOutputTensorProperties(item->properties, i);
    hbSysAllocCachedMem(&(item->sysMem[0]), item->properties.alignedByteSize);
  }
}

void BpuAsrModel::Reset() {
  offset_ = 0;
  chunk_id_ = 0;
  cached_feature_.clear();
  encoder_outs_.clear();
  encoder_outs_.resize(hidden_dim_);  // [512][0~MaxFrames]
  // Reset with zero
  for (auto& tensor : encoder_input_) {
    memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
  }
  for (auto& tensor : encoder_output_) {
    memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
  }
  for (auto& tensor : ctc_input_) {
    memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
  }
  for (auto& tensor : ctc_output_) {
    memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
  }
}

void BpuAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  // 1. Forward Encoder
  PrepareEncoderInput(chunk_feats);
  for (auto& tensor : encoder_input_) {
    TensorUtils::FlushTensor(tensor, HB_SYS_MEM_CACHE_CLEAN);
  }
  auto infer_task = task_manager_->GetModelInferTask(1000);
  infer_task->SetModel(encoder_model_);
  infer_task->SetInputTensors(encoder_input_);
  infer_task->SetOutputTensors(encoder_output_);
  infer_task->RunInfer();
  infer_task->WaitInferDone(1000);
  infer_task.reset();
  for (auto& tensor : encoder_output_) {
    TensorUtils::FlushTensor(tensor, HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // 2. Forward CTC
  PrepareCtcInput();
  for (auto& tensor : ctc_input_) {
    TensorUtils::FlushTensor(tensor, HB_SYS_MEM_CACHE_CLEAN);
  }
  infer_task = task_manager_->GetModelInferTask(1000);
  infer_task->SetModel(ctc_model_);
  infer_task->SetInputTensors(ctc_input_);
  infer_task->SetOutputTensors(ctc_output_);
  infer_task->RunInfer();
  infer_task->WaitInferDone(1000);
  infer_task.reset();
  for (auto& tensor : ctc_output_) {
    TensorUtils::FlushTensor(tensor, HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // 3. Extract final outout_prob
  const float* raw_data =
      reinterpret_cast<float*>(ctc_output_[0]->sysMem[0].virAddr);
  out_prob->resize(chunk_size_);  // v[16][4233]
  for (auto& val : *out_prob) {
    val.clear();
    val.reserve(output_dim);
  }
  for (size_t idx = 0, i = 0; i < sos_ + 1; ++i) {
    for (size_t j = 0; j < chunk_size_; ++j) {
      (*out_prob)[j].emplace_back(raw_data[idx++]);
    }
  }

  // TODO(xcsong): Suport decoder.
  //  update encoder_outs_ here.
}

void BpuAsrModel::PrepareEncoderInput(
    const std::vector<std::vector<float>>& chunk_feats) {
  chunk_id_ += 1;
  // 1. input-0: chunk
  auto& chunk = encoder_input_[0];
  auto feat_ptr = reinterpret_cast<float*>(chunk->sysMem[0].virAddr);
  memset(feat_ptr, 0, chunk->properties.alignedByteSize)
  int64_t addr_shift = 0;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {  // copy cached_feature_
    memcpy(feat_ptr + addr_shift, cached_feature_[i].data(),
           cached_feature_[i].size() * sizeof(float));
    addr_shift += cached_feature_[i].size();
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {      // copy chunk_feats
    memcpy(feat_ptr + addr_shift, chunk_feats[i].data(),
           chunk_feats[i].size() * sizeof(float));
    addr_shift += chunk_feats[i].size();
  }

  // 2. att_cache & cnn_cache
  memcpy(encoder_input_[2]->sysMem[0].virAddr,
         encoder_output_[1]->sysMem[0].virAddr,
         encoder_output_[1]->properties.alignedByteSize);
  memcpy(encoder_input_[3]->sysMem[0].virAddr,
         encoder_output_[2]->sysMem[0].virAddr,
         encoder_output_[2]->properties.alignedByteSize);

  // 3. att_mask
}

void BpuAsrModel::PrepareCtcInput() {
  // 1. chunk_out
  memcpy(ctc_input_[0]->sysMem[0].virAddr,
         encoder_output_[0]->sysMem[0].virAddr,
         encoder_output_[0]->properties.alignedByteSize);
}

float BpuAsrModel::ComputeAttentionScore(const float* prob,
                                         const std::vector<int>& hyp, int eos,
                                         int decode_out_len) {
  // TODO(xcsong): Support decoder.
  //  Currently, running decoder on edge-devices is time-consuming since the
  //  the length of input is much longer than encoder. To achieve a better
  //  accuracy-speed trade-off, we disable rescoring by default.
  return 0.0;
}

void BpuAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                     float reverse_weight,
                                     std::vector<float>* rescoring_score) {
  // TODO(xcsong): Support decoder.
  //  Currently, running decoder on edge-devices is time-consuming since the
  //  the length of input is much longer than encoder. To achieve a better
  //  accuracy-speed trade-off, we disable rescoring by default.
  LOG(INFO) << "Skip rescore.";
}

BpuAsrModel::~BpuAsrModel() {
  for (auto& tensor : encoder_input_) { hbSysFreeMem(tensor->sysMem); }
  for (auto& tensor : encoder_output_) { hbSysFreeMem(tensor->sysMem); }
  for (auto& tensor : ctc_input_) { hbSysFreeMem(tensor->sysMem); }
  for (auto& tensor : ctc_output_) { hbSysFreeMem(tensor->sysMem); }
}
}  // namespace wenet
