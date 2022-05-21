// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)
//         lizexuan@huya.com (Zexuan Li)
//         sxc19@mails.tsinghua.edu.cn (Xingchen Song)
//         hamddct@gmail.com (Mddct)

#ifndef DECODER_ONNX_ASR_MODEL_H_
#define DECODER_ONNX_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

class OnnxAsrModel : public AsrModel {
 public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);

 public:
  OnnxAsrModel() = default;
  OnnxAsrModel(const OnnxAsrModel& other);
  void Read(const std::string& model_dir);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;
  void GetInputOutputInfo(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);

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

  // sessions
  // NOTE(Mddct): The Env holds the logging state used by all other objects.
  //  One Env must be created before using any other Onnxruntime functionality.
  static Ort::Env env_;  // shared environment across threads.
  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
  std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
  std::shared_ptr<Ort::Session> ctc_session_ = nullptr;

  // node names
  std::vector<const char*> encoder_in_names_, encoder_out_names_;
  std::vector<const char*> ctc_in_names_, ctc_out_names_;
  std::vector<const char*> rescore_in_names_, rescore_out_names_;

  // caches
  Ort::Value att_cache_ort_{nullptr};
  Ort::Value cnn_cache_ort_{nullptr};
  std::vector<Ort::Value> encoder_outs_;
  // NOTE: Instead of making a copy of the xx_cache, ONNX only maintains
  //  its data pointer when initializing xx_cache_ort (see https://github.com/
  //  microsoft/onnxruntime/blob/master/onnxruntime/core/framework
  //  /tensor.cc#L102-L129), so we need the following variables to keep
  //  our data "alive" during the lifetime of decoder.
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
};

}  // namespace wenet

#endif  // DECODER_ONNX_ASR_MODEL_H_
