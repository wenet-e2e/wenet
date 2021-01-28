// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <chrono>
#include <iomanip>

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
DEFINE_int32(num_threads, 1, "num threads for device");
DEFINE_string(model_path, "", "pytorch exported model path");
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
  wenet::FeaturePipelineConfig feature_config;
  feature_config.num_bins = FLAGS_num_bins;

  std::ofstream result;
  result.open(FLAGS_result);

  std::ifstream wav_scp(FLAGS_wav_scp);
  std::string line;
  unsigned int total_waves_dur = 0;
  unsigned int total_decode_time = 0;
  while (getline(wav_scp, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    CHECK_GE(strs.size(), 2);
    std::string utt = strs[0];
    std::string wav = strs[1];

    wenet::WavReader wav_reader(wav);
    const int sample_rate = 16000;
    CHECK_EQ(wav_reader.sample_rate(), sample_rate);

    auto feature_pipeline =
        std::make_shared<wenet::FeaturePipeline>(feature_config);
    feature_pipeline->AcceptWaveform(std::vector<float>(
        wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
    feature_pipeline->set_input_finished();
    LOG(INFO) << "num frames " << feature_pipeline->num_frames();

    wenet::TorchAsrDecoder decoder(feature_pipeline, model, symbol_table,
                                   decode_config);

    unsigned int wave_dur = wav_reader.num_sample() / sample_rate * 1000;
    unsigned int decode_time = 0;
    while (true) {
      auto start = std::chrono::steady_clock::now();
      bool finish = decoder.Decode();
      auto end = std::chrono::steady_clock::now();
      decode_time +=
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      LOG(INFO) << "Partial result: " << decoder.result();
      if (finish) {
        break;
      }
    }
    LOG(INFO) << "Final result: " << decoder.result();
    LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
              << "ms.";
    result << utt << " " << decoder.result() << std::endl;

    total_waves_dur += wave_dur;
    total_decode_time += decode_time;
  }
  LOG(INFO) << "Total: decoded " << total_waves_dur << "ms audio taken "
            << total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(total_decode_time) / total_waves_dur;
  return 0;
}
