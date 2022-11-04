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

  void ForwardEncoder(
      const batch_feature_t& batch_feats,
      const std::vector<int>& batch_feats_lens,
      std::vector<std::vector<std::vector<float>>>& batch_topk_scores,
      std::vector<std::vector<std::vector<int32_t>>>& batch_topk_indexs) override;  // NOLINT

 private:
  std::shared_ptr<TorchModule> model_ = nullptr;
  torch::Tensor encoder_out_;
  torch::Tensor encoder_lens_;
  torch::DeviceType device_;
};

}  // namespace wenet

#endif  // DECODER_BATCH_TORCH_ASR_MODEL_H_
