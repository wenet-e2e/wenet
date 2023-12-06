// Copyright (c) 2022 Horizon Inc, Xingchen Song(sxc19@mails.tsinghua.edu.cn)
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

#ifndef RUNTIME_HORIZONBPU_BPU_BPU_ASR_MODEL_H_
#define RUNTIME_HORIZONBPU_BPU_BPU_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "easy_dnn/data_structure.h"
#include "easy_dnn/model.h"
#include "easy_dnn/model_manager.h"
#include "easy_dnn/task_manager.h"

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

using hobot::easy_dnn::DNNTensor;
using hobot::easy_dnn::Model;
using hobot::easy_dnn::ModelManager;
using hobot::easy_dnn::TaskManager;

namespace wenet {

class BPUAsrModel : public AsrModel {
 public:
  BPUAsrModel() = default;
  ~BPUAsrModel();
  BPUAsrModel(const BPUAsrModel& other);
  void Read(const std::string& model_dir);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;
  static void AllocMemory(const std::shared_ptr<Model>& model,
                          std::vector<std::shared_ptr<DNNTensor>>* input,
                          std::vector<std::shared_ptr<DNNTensor>>* output);
  void GetInputOutputInfo(
      const std::vector<std::shared_ptr<DNNTensor>>& input_tensors,
      const std::vector<std::shared_ptr<DNNTensor>>& output_tensors);
  void PrepareEncoderInput(const std::vector<std::vector<float>>& chunk_feats);
  void PrepareCtcInput();

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

 private:
  // metadatas
  int hidden_dim_ = 512;
  int chunk_id_ = 0;

  // models
  std::shared_ptr<Model> encoder_model_ = nullptr;
  std::shared_ptr<Model> ctc_model_ = nullptr;

  // input/output tensors
  std::vector<std::shared_ptr<DNNTensor>> encoder_input_, encoder_output_;
  std::vector<std::shared_ptr<DNNTensor>> ctc_input_, ctc_output_;
  std::vector<std::vector<float>> encoder_outs_;
};

}  // namespace wenet

#endif  // RUNTIME_HORIZONBPU_BPU_BPU_ASR_MODEL_H_
