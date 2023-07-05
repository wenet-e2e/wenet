// Copyright (c) 2023 Ximalaya Speech Team (Xiang Lyu)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frontend/wav.h"
#include "http/http_client.h"
#include "utils/flags.h"
#include "utils/timer.h"

DEFINE_string(hostname, "127.0.0.1", "hostname of http server");
DEFINE_int32(port, 10086, "port of http server");
DEFINE_int32(nbest, 1, "n-best of decode result");
DEFINE_string(wav_path, "", "test wav file path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wenet::WavReader wav_reader(FLAGS_wav_path);
  const int sample_rate = 16000;
  // Only support 16K
  CHECK_EQ(wav_reader.sample_rate(), sample_rate);
  const int num_samples = wav_reader.num_samples();
  // Convert to short
  std::vector<int16_t> data;
  data.reserve(num_samples);
  for (int j = 0; j < num_samples; j++) {
    data.push_back(static_cast<int16_t>(wav_reader.data()[j]));
  }
  // Send data
  wenet::HttpClient client(FLAGS_hostname, FLAGS_port);
  client.set_nbest(FLAGS_nbest);
  wenet::Timer timer;
  VLOG(2) << "Send " << data.size() << " samples";
  client.SendBinaryData(data.data(), data.size() * sizeof(int16_t));
  VLOG(2) << "Total latency: " << timer.Elapsed() << "ms.";
  return 0;
}
