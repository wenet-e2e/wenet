// Copyright (c) 2022  Binbin Zhang (binbzha@qq.com)
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

#include "api/batch_recognizer.h"
#include "api/wenet_api.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/timer.h"

DEFINE_string(model_dir, "", "model dir path");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_int32(batch_size, 1, "batch size of input");
DEFINE_int32(num_threads, 1, "number threads of intraop");
DEFINE_bool(enable_timestamp, false, "enable timestamps");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wenet_set_log_level(2);

  BatchRecognizer br(FLAGS_model_dir, FLAGS_num_threads);
  if (FLAGS_enable_timestamp) br.set_enable_timestamp(true);
  wenet::WavReader wav_reader(FLAGS_wav_path);
  std::vector<float> data;
  data.insert(data.end(), wav_reader.data(), wav_reader.data() + wav_reader.num_samples());
  std::vector<std::vector<float>> wavs;
  for (size_t i = 0; i < FLAGS_batch_size - 1; i++) {
    wavs.push_back(data);
  }
  wavs.push_back(std::move(data));
  wenet::Timer timer;
  std::string result = br.DecodeData(wavs);
  int forward_time = timer.Elapsed();
  VLOG(1) << "Decode() takes " << forward_time << " ms";
  LOG(INFO) << result;
  return 0;
}
