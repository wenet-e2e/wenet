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



#include <iomanip>
#include <utility>
#include <thread>
#include <sstream>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/log.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(text, "", "input text file");
DEFINE_string(confidence_file, "", "confidence output file");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitDecodeResourceFromFlags();

  // Read wav list
  std::vector<std::pair<std::string, std::string>> waves;
  CHECK(!FLAGS_wav_scp.empty());
  std::ifstream wav_scp(FLAGS_wav_scp);
  std::string line;
  while (getline(wav_scp, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    CHECK_GE(strs.size(), 2);
    waves.emplace_back(make_pair(strs[0], strs[1]));
  }

  // Read label list
  std::unordered_map<std::string, std::vector<std::string>> texts;
  CHECK(!FLAGS_text.empty());
  std::ifstream text_fs(FLAGS_text);
  while (getline(text_fs, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    CHECK_GE(strs.size(), 2);
    texts[strs[0]] = std::vector<std::string>(strs.begin() + 1, strs.end());
  }

  std::ofstream os;
  os.open(FLAGS_confidence_file, std::ios::out);
  // Compute confidence of each
  for (auto &wav : waves) {
    wenet::WavReader wav_reader(wav.second);
    CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);
    auto feature_pipeline =
        std::make_shared<wenet::FeaturePipeline>(*feature_config);
    feature_pipeline->AcceptWaveform(wav_reader.data(),
                                     wav_reader.num_samples());
    feature_pipeline->set_input_finished();
    wenet::AsrDecoder decoder(feature_pipeline, decode_resource,
                              *decode_config);
    CHECK(texts.find(wav.first) != texts.end());
    std::vector<std::string>& text = texts[wav.first];
    std::vector<float> confidence;
    decoder.ComputeTextConfidence(text, &confidence);
    CHECK_EQ(text.size(), confidence.size());
    float utt_confidence = 0.0;
    CHECK_GT(confidence.size(), 0);
    for (const auto &f : confidence) {
      utt_confidence += f;
    }
    utt_confidence /= confidence.size();
    std::stringstream ss;
    ss << wav.first << " " << utt_confidence;
    for (int i = 0; i < text.size(); i++) {
      ss << " " << text[i] << "#" << confidence[i];
    }
    LOG(INFO) << ss.str();
    ss << std::endl;
    os << ss.str();
  }
  return 0;
}
