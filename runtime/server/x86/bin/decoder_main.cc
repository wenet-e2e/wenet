// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <torch/torch.h>
#include <torch/script.h>

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_model.h"
#include "decoder/torch_asr_decoder.h"
#include "frontend/wav.h"
#include "frontend/feature_pipeline.h"

DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(chunk_size, 16, "num mel bins for fbank feature");
DEFINE_string(model_path, "", "pytorch exported model path");
DEFINE_string(wav_path, "", "wav path");
DEFINE_string(dict_path, "", "dict path");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wenet::WavReader wav_reader(FLAGS_wav_path);
  wenet::FeaturePipelineConfig feature_config;
  feature_config.num_bins = FLAGS_num_bins;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline(
        new wenet::FeaturePipeline(feature_config));
  feature_pipeline->AcceptWaveform(std::vector<float>(wav_reader.Data(),
      wav_reader.Data() +  wav_reader.NumSample()));
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->NumFramesReady();

  std::shared_ptr<wenet::TorchAsrModel> model(new wenet::TorchAsrModel);
  model->Read(FLAGS_model_path);

  wenet::SymbolTable symbol_table(FLAGS_dict_path);
  wenet::DecodeOptions decode_config;
  decode_config.chunk_size = FLAGS_chunk_size;
  wenet::TorchAsrDecoder decoder(feature_pipeline, model,
                                 symbol_table, decode_config);

  while (true) {
    bool finish = decoder.Decode();
    if (finish) {
      LOG(INFO) << "Final result: " << decoder.result();
      break;
    } else {
      LOG(INFO) << "Partial result: " << decoder.result();
    }
  }
  return 0;
}
