// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#include <iomanip>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_bool(output_nbest, false, "output n-best of decode result");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(result, "", "result output file");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitDecodeResourceFromFlags();

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
  for (auto& wav : waves) {
    wenet::WavReader wav_reader(wav.second);
    int num_samples = wav_reader.num_samples();
    CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

    auto feature_pipeline =
        std::make_shared<wenet::FeaturePipeline>(*feature_config);
    feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
    feature_pipeline->set_input_finished();
    LOG(INFO) << "num frames " << feature_pipeline->num_frames();

    wenet::AsrDecoder decoder(feature_pipeline, decode_resource,
                              *decode_config);

    int wave_dur = static_cast<int>(static_cast<float>(num_samples) /
                                    wav_reader.sample_rate() * 1000);
    int decode_time = 0;
    std::string final_result;
    while (true) {
      wenet::Timer timer;
      wenet::DecodeState state = decoder.Decode();
      if (state == wenet::DecodeState::kEndFeats) {
        decoder.Rescoring();
      }
      int chunk_decode_time = timer.Elapsed();
      decode_time += chunk_decode_time;
      if (decoder.DecodedSomething()) {
        LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
      }

      if (FLAGS_continuous_decoding && state == wenet::DecodeState::kEndpoint) {
        if (decoder.DecodedSomething()) {
          decoder.Rescoring();
          final_result.append(decoder.result()[0].sentence);
        }
        decoder.ResetContinuousDecoding();
      }

      if (state == wenet::DecodeState::kEndFeats) {
        break;
      } else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
        float frame_shift_in_ms =
            static_cast<float>(feature_config->frame_shift) /
            wav_reader.sample_rate() * 1000;
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
    if (decoder.DecodedSomething()) {
      final_result.append(decoder.result()[0].sentence);
    }
    LOG(INFO) << wav.first << " Final result: " << final_result << std::endl;
    LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
              << "ms.";

    if (!FLAGS_output_nbest) {
      buffer << wav.first << " " << final_result << std::endl;
    } else {
      buffer << "wav " << wav.first << std::endl;
      auto& results = decoder.result();
      for (auto& r : results) {
        if (r.sentence.empty()) continue;
        buffer << "candidate " << r.score << " " << r.sentence << std::endl;
      }
    }

    total_waves_dur += wave_dur;
    total_decode_time += decode_time;
  }
  LOG(INFO) << "Total: decoded " << total_waves_dur << "ms audio taken "
            << total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(total_decode_time) / total_waves_dur;
  return 0;
}
