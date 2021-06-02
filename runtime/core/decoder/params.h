// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

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
DEFINE_int32(sample_rate, 16000, "sample rate for audio");

// TLG fst
DEFINE_string(fst_path, "", "TLG fst path");

// DecodeOptions flags
DEFINE_int32(chunk_size, 16, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_double(ctc_weight, 0.0,
              "ctc weight when combining ctc score and rescoring score");
DEFINE_double(rescoring_weight, 1.0,
              "rescoring weight when combining ctc score and rescoring score");
DEFINE_double(reverse_weight, 0.0,
              "used for bitransformer rescoring. it must be 0.0 if decoder is"
              "conventional transformer decoder, and only reverse_weight > 0.0"
              "dose the right to left decoder will be calculated and used");
DEFINE_int32(max_active, 7000, "max active states in ctc wfst search");
DEFINE_double(beam, 16.0, "beam in ctc wfst search");
DEFINE_double(lattice_beam, 10.0, "lattice beam in ctc wfst search");
DEFINE_double(acoustic_scale, 1.0, "acoustic scale for ctc wfst search");
DEFINE_double(blank_skip_thresh, 1.0,
              "blank skip thresh for ctc wfst search, 1.0 means no skip");

// SymbolTable flags
DEFINE_string(dict_path, "", "dict path");

namespace wenet {

std::shared_ptr<TorchAsrModel> InitTorchAsrModelFromFlags() {
  auto model = std::make_shared<TorchAsrModel>();
  model->Read(FLAGS_model_path, FLAGS_num_threads);
  return model;
}

std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  auto feature_config = std::make_shared<FeaturePipelineConfig>(FLAGS_num_bins,
    FLAGS_sample_rate);
  return feature_config;
}

std::shared_ptr<fst::Fst<fst::StdArc>> InitFstFromFlags() {
  std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
  if (!FLAGS_fst_path.empty()) {
    fst.reset(fst::Fst<fst::StdArc>::Read(FLAGS_fst_path));
    CHECK(fst != nullptr);
  }
  return fst;
}

std::shared_ptr<DecodeOptions> InitDecodeOptionsFromFlags() {
  auto decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = FLAGS_chunk_size;
  decode_config->num_left_chunks = FLAGS_num_left_chunks;
  decode_config->ctc_weight = FLAGS_ctc_weight;
  decode_config->reverse_weight = FLAGS_reverse_weight;
  decode_config->rescoring_weight = FLAGS_rescoring_weight;
  decode_config->ctc_wfst_search_opts.max_active = FLAGS_max_active;
  decode_config->ctc_wfst_search_opts.beam = FLAGS_beam;
  decode_config->ctc_wfst_search_opts.lattice_beam = FLAGS_lattice_beam;
  decode_config->ctc_wfst_search_opts.acoustic_scale = FLAGS_acoustic_scale;
  decode_config->ctc_wfst_search_opts.blank_skip_thresh =
      FLAGS_blank_skip_thresh;
  return decode_config;
}

std::shared_ptr<fst::SymbolTable> InitSymbolTableFromFlags() {
  auto symbol_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_dict_path));
  return symbol_table;
}

}  // namespace wenet

#endif  // DECODER_PARAMS_H_
