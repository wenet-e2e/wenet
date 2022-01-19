// Copyright 2021 Huya Inc. All Rights Reserved.
// Author: lizexuan@huya.com (Zexuan Li)

#include "decoder/onnx_asr_model.h"

#include <utility>
#include <memory>
#include <string>

namespace wenet {

void OnnxAsrModel::Read(const std::string &model_dir) {
  std::string encoder_onnx_path = model_dir + "/encoder.onnx";
  std::string rescore_onnx_path = model_dir + "/rescore.onnx";
  std::string ctc_onnx_path = model_dir + "/ctc.onnx";
  std::string onnx_conf_path = model_dir + "/onnx.conf";

  try {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

    Ort::Session encoder_session{env_, encoder_onnx_path.data(),
                                 session_options};
    encoder_session_ = std::make_shared<Ort::Session>(
                            std::move(encoder_session));

    Ort::Session rescore_session{env_, rescore_onnx_path.data(),
                                 session_options};
    rescore_session_ = std::make_shared<Ort::Session>(
                            std::move(rescore_session));

    Ort::Session ctc_session{env_, ctc_onnx_path.data(), session_options};
    ctc_session_ = std::make_shared<Ort::Session>(std::move(ctc_session));
  } catch (std::exception const &e) {
    printf("%s", e.what());
    exit(0);
  }

  std::ifstream infile;

  infile.open(onnx_conf_path);

  if (!infile) {
    printf("error when open %s\n", onnx_conf_path.data());
    exit(0);
  }

  std::string data;

  infile >> data;
  encoder_output_size_ = std::stoi(data);

  infile >> data;
  num_blocks_ = std::stoi(data);

  infile >> data;
  cnn_module_kernel_ = std::stoi(data);

  infile >> data;
  subsampling_rate_ = std::stoi(data);

  infile >> data;
  right_context_ = std::stoi(data);

  infile >> data;
  sos_ = std::stoi(data);

  infile >> data;
  eos_ = std::stoi(data);

  infile >> data;
  is_bidirectional_decoder_ = std::stoi(data);

  infile.close();
}

}  // namespace wenet
