// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#ifndef DECODER_TORCH_ASR_MODEL_H_
#define DECODER_TORCH_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/asr_model.h"
#include "utils/utils.h"

namespace wenet {

class TorchAsrModel : public AsrModel {
 public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);

 public:
  using TorchModule = torch::jit::script::Module;
  TorchAsrModel() = default;
  TorchAsrModel(const TorchAsrModel& other);
  void Read(const std::string& model_path);
  std::shared_ptr<TorchModule> torch_model() const { return model_; }
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const torch::Tensor& prob,
                              const std::vector<int>& hyp, int eos);

 private:
  std::shared_ptr<TorchModule> model_ = nullptr;
  std::vector<torch::Tensor> encoder_outs_;
  // transformer/conformer attention cache
  torch::Tensor att_cache_ = torch::zeros({0, 0, 0, 0});
  // conformer-only conv_module cache
  torch::Tensor cnn_cache_ = torch::zeros({0, 0, 0, 0});
};

}  // namespace wenet

#endif  // DECODER_TORCH_ASR_MODEL_H_
