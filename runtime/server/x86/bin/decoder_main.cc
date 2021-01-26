// Copyright (c) 2021 Mobvoi Inc (authors: Zhendong Peng)
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
#include <chrono>
#include <string>

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "utils/cxxopts.h"

int main(int argc, char *argv[]) {
  cxxopts::Options options("compute_rtf", "Compute the RTF of wenet");
  options.add_options()
      ("n,num_bins", "num mel bins for fbank feature",
          cxxopts::value<int>()->default_value("80"))
      ("c,chunk_size", "decoding chunk size",
          cxxopts::value<int>()->default_value("16"))
      ("m,model_path", "pytorch exported model path",
          cxxopts::value<std::string>())
      ("d,dict_path", "dict path", cxxopts::value<std::string>())(
      "s,scp_path", "wav scp path", cxxopts::value<std::string>());
  auto args = options.parse(argc, argv);

  int num_bins = args["num_bins"].as<int>();
  int chunk_size = args["chunk_size"].as<int>();
  std::string model_path = args["model_path"].as<std::string>();
  std::string dict_path = args["dict_path"].as<std::string>();
  std::string wav_scp = args["scp_path"].as<std::string>();

  auto model = std::make_shared<wenet::TorchAsrModel>();
  model->Read(model_path);
  wenet::SymbolTable symbol_table(dict_path);
  wenet::DecodeOptions decode_config;
  decode_config.chunk_size = chunk_size;
  wenet::FeaturePipelineConfig feature_config;
  feature_config.num_bins = num_bins;

  std::ifstream scp(wav_scp);
  std::string wav;
  unsigned int waves_dur = 0;
  unsigned int decode_time = 0;
  while (getline(scp, wav)) {
    wenet::WavReader wav_reader(wav);
    auto feature_pipeline =
        std::make_shared<wenet::FeaturePipeline>(feature_config);
    feature_pipeline->AcceptWaveform(std::vector<float>(
        wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
    feature_pipeline->set_input_finished();
    wenet::TorchAsrDecoder decoder(feature_pipeline, model, symbol_table,
                                   decode_config);

    int dur = wav_reader.num_sample() / 16;
    auto start = std::chrono::steady_clock::now();
    while (true) {
      bool finish = decoder.Decode();
      if (finish) {
        LOG(INFO) << "Final result: " << decoder.result();
        break;
      } else {
        LOG(INFO) << "Partial result: " << decoder.result();
      }
    }
    auto end = std::chrono::steady_clock::now();
    int time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    LOG(INFO) << wav << " " << dur << "ms, taken " << time << "ms.";
    waves_dur += dur;
    decode_time += time;
  }
  LOG(INFO) << "Total" << waves_dur << "ms, taken " << decode_time << "ms.";
  LOG(INFO) << "RTF: " << static_cast<float>(decode_time) / waves_dur;
  return 0;
}
