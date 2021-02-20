// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <chrono>
#include <iomanip>
#include <utility>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "utils/utils.h"

DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(chunk_size, 16, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_int32(num_threads, 1, "num threads for device");
DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_string(model_path, "", "pytorch exported model path");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(dict_path, "", "dict path");
DEFINE_string(result, "", "result output file");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto model = std::make_shared<wenet::TorchAsrModel>();
  model->Read(FLAGS_model_path, FLAGS_num_threads);
  wenet::SymbolTable symbol_table(FLAGS_dict_path);
  wenet::DecodeOptions decode_config;
  decode_config.chunk_size = FLAGS_chunk_size;
  decode_config.num_left_chunks = FLAGS_num_left_chunks;
  wenet::FeaturePipelineConfig feature_config;
  feature_config.num_bins = FLAGS_num_bins;
  const int sample_rate = 16000;
  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(feature_config);

  if (FLAGS_wav_path.empty() && FLAGS_wav_scp.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp.";
  }
  std::vector<std::pair<std::string, std::string>> waves;
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wenet::SplitString(line, &strs);
      CHECK_GE(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }
  }

  std::ofstream result;
  if (!FLAGS_result.empty()) {
    result.open(FLAGS_result, std::ios::out);
  }
  std::ostream& buffer = FLAGS_result.empty() ? std::cout : result;

  int total_waves_dur = 0;
  int total_decode_time = 0;
  for (auto &wav : waves) {
    wenet::WavReader wav_reader(wav.second);
    CHECK_EQ(wav_reader.sample_rate(), sample_rate);

    feature_pipeline->AcceptWaveform(std::vector<float>(
        wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
    feature_pipeline->set_input_finished();
    LOG(INFO) << "num frames " << feature_pipeline->num_frames();

    wenet::TorchAsrDecoder decoder(feature_pipeline, model, symbol_table,
                                   decode_config);

    int wave_dur = wav_reader.num_sample() / sample_rate * 1000;
    int decode_time = 0;
    while (true) {
      auto start = std::chrono::steady_clock::now();
      bool finish = decoder.Decode();
      auto end = std::chrono::steady_clock::now();
      auto chunk_decode_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      decode_time += chunk_decode_time;
      LOG(INFO) << "Partial result: " << decoder.result();

      if (finish) {
        break;
      } else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
        float frame_shift_in_ms =
            static_cast<float>(feature_config.frame_shift) / sample_rate * 1000;
        auto wait_time =
            decoder.num_frames_in_current_chunk() * frame_shift_in_ms -
            chunk_decode_time;
        if (wait_time > 0) {
          LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
          std::this_thread::sleep_for(
              std::chrono::milliseconds(static_cast<int>(wait_time)));
        }
      }
    }
    LOG(INFO) << "Final result: " << decoder.result();
    LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
              << "ms.";
    buffer << wav.first << " " << decoder.result() << std::endl;

    total_waves_dur += wave_dur;
    total_decode_time += decode_time;
  }
  LOG(INFO) << "Total: decoded " << total_waves_dur << "ms audio taken "
            << total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(total_decode_time) / total_waves_dur;
  return 0;
}
