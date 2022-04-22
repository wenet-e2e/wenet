// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
// Copyright 2022 Mobvoi Inc. All Rights Reserved.
// Author: lizexuan@huya.com

#ifndef DECODER_ONNX_ASR_MODEL_H_
#define DECODER_ONNX_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "decoder/asr_model.h"
#include "utils/utils.h"

namespace wenet {

class OnnxAsrModel : public AsrModel {
 public:
  using TorchModule = torch::jit::script::Module;
  OnnxAsrModel() = default;
  OnnxAsrModel(const OnnxAsrModel& other);
  void Read(const std::string& model_dir, int num_threads = 1);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

 private:
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;

  Ort::Env env_;
  std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
  std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
  std::shared_ptr<Ort::Session> ctc_session_ = nullptr;

  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const char* input_names_[6] = {
      "chunk",     "offset",    "required_cache_size",
      "att_cache", "cnn_cache", "att_mask"};
  const char* output_names_[3] = {"output", "r_att_cache", "r_cnn_cache"};
  const char* decode_input_names_[3] = {"hyps_pad", "hyps_lens", "encoder_out"};
  const char* decode_output_names_[2] = {"o1", "o2"};

  const char* ctc_input_names_[1] = {"hidden"};
  const char* ctc_output_names_[1] = {"probs"};

  Ort::Value att_cache_ort_{nullptr};
  Ort::Value cnn_cache_ort_{nullptr};
  Ort::Value att_mask_ort_{nullptr};
  std::vector<Ort::Value> encoder_outs_;
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
  std::vector<uint8_t> att_mask_;
};

}  // namespace wenet

#endif  // DECODER_ONNX_ASR_MODEL_H_
