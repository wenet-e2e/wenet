// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <torch/torch.h>
#include <torch/script.h>

#include "decoder/torch_asr_model.h"

DEFINE_string(model_path, "", "pytorch exported model path");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << FLAGS_model_path;

  wenet::TorchAsrModel model;
  model.Read(FLAGS_model_path);
  return 0;
}
