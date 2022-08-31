// Copyright (c) 2022 SDCI Co. Ltd (author: veelion)
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


#ifndef DECODER_BATCH_TORCH_ASR_MODEL_H_
#define DECODER_BATCH_TORCH_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/batch_asr_model.h"
#include "utils/utils.h"

namespace wenet {

class BatchTorchAsrModel : public BatchAsrModel {
public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);

 public:
  using TorchModule = torch::jit::script::Module;
  BatchTorchAsrModel() = default;
  BatchTorchAsrModel(const BatchTorchAsrModel& other);
  void Read(const std::string& model_path);
  void AttentionRescoring(
      const std::vector<std::vector<std::vector<int>>>& batch_hyps,
      const std::vector<std::vector<float>>& ctc_scores,
      std::vector<std::vector<float>>& attention_scores) override;
  std::shared_ptr<BatchAsrModel> Copy() const override;

 protected:
  void ForwardEncoderFunc(
      const batch_feature_t& batch_feats,
      const std::vector<int>& batch_feats_lens,
      batch_ctc_log_prob_t& batch_ctc_log_prob) override;

  float ComputeAttentionScore(const torch::Tensor& batch_prob,
                              const std::vector<int>& hyp, int eos);

 private:
  std::shared_ptr<TorchModule> model_ = nullptr;
  torch::Tensor encoder_out_;
  torch::Tensor encoder_lens_;
  torch::DeviceType device_;
};

}  // namespace wenet

#endif  // DECODER_BATCH_TORCH_ASR_MODEL_H_
