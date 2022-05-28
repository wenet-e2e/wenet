// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#ifndef DECODER_PARAMS_H_
#define DECODER_PARAMS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "decoder/asr_decoder.h"
#ifdef USE_ONNX
#include "decoder/onnx_asr_model.h"
#endif
#ifdef USE_TORCH
#include "decoder/torch_asr_model.h"
#endif
#include "frontend/feature_pipeline.h"
#include "post_processor/post_processor.h"
#include "utils/flags.h"
#include "utils/string.h"

DEFINE_int32(num_threads, 1, "num threads for ASR model");

// TorchAsrModel flags
DEFINE_string(model_path, "", "pytorch exported model path");
// OnnxAsrModel flags
DEFINE_string(onnx_dir, "", "directory where the onnx model is saved");

// FeaturePipelineConfig flags
DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(sample_rate, 16000, "sample rate for audio");

// TLG fst
DEFINE_string(fst_path, "", "TLG fst path");

// DecodeOptions flags
DEFINE_int32(chunk_size, 16, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_double(ctc_weight, 0.5,
              "ctc weight when combining ctc score and rescoring score");
DEFINE_double(rescoring_weight, 1.0,
              "rescoring weight when combining ctc score and rescoring score");
DEFINE_double(reverse_weight, 0.0,
              "used for bitransformer rescoring. it must be 0.0 if decoder is"
              "conventional transformer decoder, and only reverse_weight > 0.0"
              "dose the right to left decoder will be calculated and used");
DEFINE_int32(max_active, 7000, "max active states in ctc wfst search");
DEFINE_int32(min_active, 200, "min active states in ctc wfst search");
DEFINE_double(beam, 16.0, "beam in ctc wfst search");
DEFINE_double(lattice_beam, 10.0, "lattice beam in ctc wfst search");
DEFINE_double(acoustic_scale, 1.0, "acoustic scale for ctc wfst search");
DEFINE_double(blank_skip_thresh, 1.0,
              "blank skip thresh for ctc wfst search, 1.0 means no skip");
DEFINE_double(length_penalty, 0.0, "length penalty ctc wfst search, will not"
              "apply on self-loop arc, for balancing the del/ins ratio, "
              "suggest set to -3.0");
DEFINE_int32(nbest, 10, "nbest for ctc wfst or prefix search");

// SymbolTable flags
DEFINE_string(dict_path, "",
              "dict symbol table path, it's same as unit_path when we don't "
              "use LM in decoding");
DEFINE_string(
    unit_path, "",
    "e2e model unit symbol table, is used to get timestamp of the result");

// Context flags
DEFINE_string(context_path, "", "context path, is used to build context graph");
DEFINE_double(context_score, 3.0, "is used to rescore the decoded result");

// PostProcessOptions flags
DEFINE_int32(language_type, 0,
             "remove spaces according to language type"
             "0x00 = kMandarinEnglish, "
             "0x01 = kIndoEuropean");
DEFINE_bool(lowercase, true, "lowercase final result if needed");

namespace wenet {
std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  auto feature_config = std::make_shared<FeaturePipelineConfig>(
      FLAGS_num_bins, FLAGS_sample_rate);
  return feature_config;
}

std::shared_ptr<DecodeOptions> InitDecodeOptionsFromFlags() {
  auto decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = FLAGS_chunk_size;
  decode_config->num_left_chunks = FLAGS_num_left_chunks;
  decode_config->ctc_weight = FLAGS_ctc_weight;
  decode_config->reverse_weight = FLAGS_reverse_weight;
  decode_config->rescoring_weight = FLAGS_rescoring_weight;
  decode_config->ctc_wfst_search_opts.max_active = FLAGS_max_active;
  decode_config->ctc_wfst_search_opts.min_active = FLAGS_min_active;
  decode_config->ctc_wfst_search_opts.beam = FLAGS_beam;
  decode_config->ctc_wfst_search_opts.lattice_beam = FLAGS_lattice_beam;
  decode_config->ctc_wfst_search_opts.acoustic_scale = FLAGS_acoustic_scale;
  decode_config->ctc_wfst_search_opts.blank_skip_thresh =
      FLAGS_blank_skip_thresh;
  decode_config->ctc_wfst_search_opts.length_penalty = FLAGS_length_penalty;
  decode_config->ctc_wfst_search_opts.nbest = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.first_beam_size = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.second_beam_size = FLAGS_nbest;
  return decode_config;
}

std::shared_ptr<DecodeResource> InitDecodeResourceFromFlags() {
  auto resource = std::make_shared<DecodeResource>();

  if (!FLAGS_onnx_dir.empty()) {
#ifdef USE_ONNX
    LOG(INFO) << "Reading onnx model ";
    OnnxAsrModel::InitEngineThreads(FLAGS_num_threads);
    auto model = std::make_shared<OnnxAsrModel>();
    model->Read(FLAGS_onnx_dir);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DONNX=ON'.";
#endif
  } else {
#ifdef USE_TORCH
    LOG(INFO) << "Reading torch model " << FLAGS_model_path;
    TorchAsrModel::InitEngineThreads(FLAGS_num_threads);
    auto model = std::make_shared<TorchAsrModel>();
    model->Read(FLAGS_model_path);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DTORCH=ON'.";
#endif
  }

  std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
  if (!FLAGS_fst_path.empty()) {
    LOG(INFO) << "Reading fst " << FLAGS_fst_path;
    fst.reset(fst::Fst<fst::StdArc>::Read(FLAGS_fst_path));
    CHECK(fst != nullptr);
  }
  resource->fst = fst;

  LOG(INFO) << "Reading symbol table " << FLAGS_dict_path;
  auto symbol_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_dict_path));
  resource->symbol_table = symbol_table;

  std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
  if (!FLAGS_unit_path.empty()) {
    LOG(INFO) << "Reading unit table " << FLAGS_unit_path;
    unit_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(FLAGS_unit_path));
    CHECK(unit_table != nullptr);
  } else if (fst == nullptr) {
    LOG(INFO) << "Use symbol table as unit table";
    unit_table = symbol_table;
  }
  resource->unit_table = unit_table;

  if (!FLAGS_context_path.empty()) {
    LOG(INFO) << "Reading context " << FLAGS_context_path;
    std::vector<std::string> contexts;
    std::ifstream infile(FLAGS_context_path);
    std::string context;
    while (getline(infile, context)) {
      contexts.emplace_back(Trim(context));
    }
    ContextConfig config;
    config.context_score = FLAGS_context_score;
    resource->context_graph = std::make_shared<ContextGraph>(config);
    resource->context_graph->BuildContextGraph(contexts, symbol_table);
  }

  PostProcessOptions post_process_opts;
  post_process_opts.language_type =
      FLAGS_language_type == 0 ? kMandarinEnglish : kIndoEuropean;
  post_process_opts.lowercase = FLAGS_lowercase;
  resource->post_processor =
      std::make_shared<PostProcessor>(std::move(post_process_opts));
  return resource;
}

}  // namespace wenet

#endif  // DECODER_PARAMS_H_
