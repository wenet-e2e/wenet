// Copyright 2021 Huya Inc. All Rights Reserved.
// Author: lizexuan@huya.com (Zexuan Li)

#ifndef DECODER_ONNX_ASR_MODEL_H_
#define DECODER_ONNX_ASR_MODEL_H_

#include <fstream>
#include <algorithm>
#include <string>
#include <memory>
#include "./onnxruntime_cxx_api.h"
#include "utils/utils.h"

namespace wenet {

class OnnxAsrModel {
 public:
  OnnxAsrModel() = default;
  void Read(const std::string& model_dir);
  int right_context() const { return right_context_; }
  int subsampling_rate() const { return subsampling_rate_; }
  int sos() const { return sos_; }
  int eos() const { return eos_; }
  int encoder_output_size() const { return encoder_output_size_; }
  int num_blocks() const { return num_blocks_; }
  int cnn_module_kernel() const { return cnn_module_kernel_; }
  bool is_bidirectional_decoder() const { return is_bidirectional_decoder_; }
  std::shared_ptr<Ort::Session> encoder_session() const {
    return encoder_session_;
  }
  std::shared_ptr<Ort::Session> rescore_session() const {
    return rescore_session_;
  }
  std::shared_ptr<Ort::Session> ctc_session() const {
    return ctc_session_;
  }

 private:
  Ort::Env env_;
  std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
  std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
  std::shared_ptr<Ort::Session> ctc_session_ = nullptr;
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int right_context_ = 1;
  int subsampling_rate_ = 1;
  int sos_ = 0;
  int eos_ = 0;
  bool is_bidirectional_decoder_ = false;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(OnnxAsrModel);
};

}  // namespace wenet

#endif  // DECODER_ONNX_ASR_MODEL_H_
