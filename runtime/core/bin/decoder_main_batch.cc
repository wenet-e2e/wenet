// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
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

#include <iomanip>
#include <thread>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_string(wav_path, "", "single wave path");
DEFINE_int32(thread_num, 1, "num of decode thread");
DEFINE_int32(batch_size, 1, "batch size of input");

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

int g_total_waves_dur = 0;
int g_total_decode_time = 0;

// using namespace wenet;

void decode(const std::string& wav) {
  wenet::WavReader wav_reader(wav);
  std::vector<float> wav_data;
  int num_samples = wav_reader.num_samples();
  wav_data.insert(
      wav_data.end(), wav_reader.data(), wav_reader.data() + num_samples);
  std::vector<std::vector<float>> batch_wav_data;
  int wav_dur = static_cast<int>(
      static_cast<float>(num_samples) / wav_reader.sample_rate() * 1000);
  for (int i = 0; i < FLAGS_batch_size; ++i) {
    batch_wav_data.push_back(wav_data);
    g_total_waves_dur += wav_dur;
  }

  auto decoder = std::make_unique<wenet::BatchAsrDecoder>(
      g_feature_config, g_decode_resource, *g_decode_config);
  wenet::Timer timer;
  decoder->Decode(batch_wav_data);
  int decode_time = timer.Elapsed();
  std::string result = decoder->get_batch_result(1, false);
  std::cout << result << std::endl;

  LOG(INFO) << "batch_size : " << FLAGS_batch_size << std::endl;
  LOG(INFO) << "Total: decoded " << g_total_waves_dur << "ms audio taken "
            << decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(decode_time) / g_total_waves_dur;
}


int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  g_decode_config = wenet::InitDecodeOptionsFromFlags();
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  g_decode_resource = wenet::InitDecodeResourceFromFlags();

  if (FLAGS_wav_path.empty()) {
    LOG(FATAL) << "Please provide the wave path.";
  }
  LOG(INFO) << "decoding " << FLAGS_wav_path;
  decode(FLAGS_wav_path);

  return 0;
}
