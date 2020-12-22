// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <torch/torch.h>
#include <torch/script.h>

DEFINE_int32(num, 10, "int help");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << FLAGS_num;
  return 0;
}
