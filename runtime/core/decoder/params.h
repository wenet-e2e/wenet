// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_PARAMS_H_
#define DECODER_PARAMS_H_

#include <memory>

#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "utils/flags.h"

// TorchAsrModel flags
DEFINE_int32(num_threads, 1, "num threads for device");
DEFINE_string(model_path, "", "pytorch exported model path");

// FeaturePipelineConfig flags
DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");

// DecodeOptions flags
DEFINE_int32(chunk_size, 16, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_double(ctc_weight, 0.0,
              "ctc weight when combining ctc score and rescoring score");
DEFINE_double(rescoring_weight, 1.0,
              "rescoring weight when combining ctc score and rescoring score");

// SymbolTable flags
DEFINE_string(dict_path, "", "dict path");

namespace wenet {

std::shared_ptr<TorchAsrModel> InitTorchAsrModelFromFlags() {
  auto model = std::make_shared<TorchAsrModel>();
  model->Read(FLAGS_model_path, FLAGS_num_threads);
  return model;
}

std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  auto feature_config = std::make_shared<FeaturePipelineConfig>();
  feature_config->num_bins = FLAGS_num_bins;
  return feature_config;
}

std::shared_ptr<DecodeOptions> InitDecodeOptionsFromFlags() {
  auto decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = FLAGS_chunk_size;
  decode_config->num_left_chunks = FLAGS_num_left_chunks;
  decode_config->ctc_weight = FLAGS_ctc_weight;
  decode_config->rescoring_weight = FLAGS_rescoring_weight;
  return decode_config;
}

std::shared_ptr<fst::SymbolTable> InitSymbolTableFromFlags() {
  auto symbol_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_dict_path));
  return symbol_table;
}

}  // namespace wenet

#endif  // DECODER_PARAMS_H_
