// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
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
#ifdef USE_XPU
#include "xpu/xpu_asr_model.h"
#endif
#ifdef USE_BPU
#include "bpu/bpu_asr_model.h"
#endif
#ifdef USE_OPENVINO
#include "ov/ov_asr_model.h"
#endif
#include "frontend/feature_pipeline.h"
#include "post_processor/post_processor.h"
#include "utils/file.h"
#include "utils/flags.h"
#include "utils/string.h"

DEFINE_int32(device_id, 0, "set XPU DeviceID for ASR model");

// TorchAsrModel flags
DEFINE_string(model_path, "", "pytorch exported model path");
// OnnxAsrModel flags
DEFINE_string(onnx_dir, "", "directory where the onnx model is saved");
// XPUAsrModel flags
DEFINE_string(xpu_model_dir, "",
              "directory where the XPU model and weights is saved");
// BPUAsrModel flags
DEFINE_string(bpu_model_dir, "",
              "directory where the HORIZON BPU model is saved");
// OVAsrModel flags
DEFINE_string(openvino_dir, "", "directory where the OV model is saved");
DEFINE_int32(core_number, 1, "Core number of process");

// FeaturePipelineConfig flags
DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(sample_rate, 16000, "sample rate for audio");
DEFINE_string(feat_type, "kaldi", "Type of feature extraction: kaldi, whisper");

// TLG fst
DEFINE_string(fst_path, "", "TLG fst path");

// ITN fst
DEFINE_string(itn_model_dir, "",
              "fst based ITN model dir, "
              "should contain itn_tagger.fst and itn_verbalizer.fst");

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
DEFINE_int32(blank_id, 0,
             "blank token idx for ctc wfst search and ctc prefix beam search");
DEFINE_double(blank_skip_thresh, 1.0,
              "blank skip thresh for ctc wfst search, 1.0 means no skip");
DEFINE_double(blank_scale, 1.0, "blank scale for ctc wfst search");
DEFINE_double(length_penalty, 0.0,
              "length penalty ctc wfst search, will not"
              "apply on self-loop arc, for balancing the del/ins ratio, "
              "suggest set to -3.0");
DEFINE_int32(nbest, 10, "nbest for ctc wfst or prefix search");

// SymbolTable flags
DEFINE_string(dict_path, "",
              "dict symbol table path, required when LM is enabled");
DEFINE_string(unit_path, "",
              "e2e model unit symbol table, it is used in both "
              "with/without LM scenarios for context/timestamp");

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

FeatureType StringToFeatureType(const std::string& feat_type_str) {
  if (feat_type_str == "kaldi")
    return FeatureType::kKaldi;
  else if (feat_type_str == "whisper")
    return FeatureType::kWhisper;
  else
    throw std::invalid_argument("Unsupported feat type!");
}

std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  FeatureType feat_type = StringToFeatureType(FLAGS_feat_type);
  auto feature_config = std::make_shared<FeaturePipelineConfig>(
      FLAGS_num_bins, FLAGS_sample_rate, feat_type);
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
  decode_config->ctc_wfst_search_opts.blank = FLAGS_blank_id;
  decode_config->ctc_wfst_search_opts.blank_skip_thresh =
      FLAGS_blank_skip_thresh;
  decode_config->ctc_wfst_search_opts.blank_scale = FLAGS_blank_scale;
  decode_config->ctc_wfst_search_opts.length_penalty = FLAGS_length_penalty;
  decode_config->ctc_wfst_search_opts.nbest = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.first_beam_size = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.second_beam_size = FLAGS_nbest;
  decode_config->ctc_prefix_search_opts.blank = FLAGS_blank_id;
  decode_config->ctc_endpoint_config.blank = FLAGS_blank_id;
  decode_config->ctc_endpoint_config.blank_scale = FLAGS_blank_scale;
  return decode_config;
}

std::shared_ptr<DecodeResource> InitDecodeResourceFromFlags() {
  auto resource = std::make_shared<DecodeResource>();
  const int kNumGemmThreads = 1;
  if (!FLAGS_onnx_dir.empty()) {
#ifdef USE_ONNX
    LOG(INFO) << "Reading onnx model ";
    OnnxAsrModel::InitEngineThreads(kNumGemmThreads);
    auto model = std::make_shared<OnnxAsrModel>();
    model->Read(FLAGS_onnx_dir);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DONNX=ON'.";
#endif
  } else if (!FLAGS_model_path.empty()) {
#ifdef USE_TORCH
    LOG(INFO) << "Reading torch model " << FLAGS_model_path;
    TorchAsrModel::InitEngineThreads(kNumGemmThreads);
    auto model = std::make_shared<TorchAsrModel>();
    model->Read(FLAGS_model_path);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DTORCH=ON'.";
#endif
  } else if (!FLAGS_xpu_model_dir.empty()) {
#ifdef USE_XPU
    LOG(INFO) << "Reading XPU WeNet model weight from " << FLAGS_xpu_model_dir;
    auto model = std::make_shared<XPUAsrModel>();
    model->SetEngineThreads(kNumGemmThreads);
    model->SetDeviceId(FLAGS_device_id);
    model->Read(FLAGS_xpu_model_dir);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DXPU=ON'.";
#endif
  } else if (!FLAGS_bpu_model_dir.empty()) {
#ifdef USE_BPU
    LOG(INFO) << "Reading Horizon BPU model from " << FLAGS_bpu_model_dir;
    auto model = std::make_shared<BPUAsrModel>();
    model->Read(FLAGS_bpu_model_dir);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DBPU=ON'.";
#endif
  } else if (!FLAGS_openvino_dir.empty()) {
#ifdef USE_OPENVINO
    LOG(INFO) << "Read OpenVINO model ";
    auto model = std::make_shared<OVAsrModel>();
    model->InitEngineThreads(FLAGS_core_number);
    model->Read(FLAGS_openvino_dir);
    resource->model = model;
#else
    LOG(FATAL) << "Please rebuild with cmake options '-DOPENVINO=ON'.";
#endif
  } else {
    LOG(FATAL) << "Please set ONNX, TORCH, XPU, BPU or OpenVINO model path!!!";
  }

  LOG(INFO) << "Reading unit table " << FLAGS_unit_path;
  auto unit_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(FLAGS_unit_path));
  CHECK(unit_table != nullptr);
  resource->unit_table = unit_table;

  if (!FLAGS_fst_path.empty()) {  // With LM
    CHECK(!FLAGS_dict_path.empty());
    LOG(INFO) << "Reading fst " << FLAGS_fst_path;
    auto fst = std::shared_ptr<fst::VectorFst<fst::StdArc>>(
        fst::VectorFst<fst::StdArc>::Read(FLAGS_fst_path));
    CHECK(fst != nullptr);
    resource->fst = fst;

    LOG(INFO) << "Reading symbol table " << FLAGS_dict_path;
    auto symbol_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(FLAGS_dict_path));
    CHECK(symbol_table != nullptr);
    resource->symbol_table = symbol_table;
  } else {  // Without LM, symbol_table is the same as unit_table
    resource->symbol_table = unit_table;
  }

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
    resource->context_graph->BuildContextGraph(contexts, unit_table);
  }

  PostProcessOptions post_process_opts;
  post_process_opts.language_type =
      FLAGS_language_type == 0 ? kMandarinEnglish : kIndoEuropean;
  post_process_opts.lowercase = FLAGS_lowercase;
  resource->post_processor =
      std::make_shared<PostProcessor>(std::move(post_process_opts));

  if (!FLAGS_itn_model_dir.empty()) {  // With ITN
    std::string itn_tagger_path =
        wenet::JoinPath(FLAGS_itn_model_dir, "zh_itn_tagger.fst");
    std::string itn_verbalizer_path =
        wenet::JoinPath(FLAGS_itn_model_dir, "zh_itn_verbalizer.fst");
    if (wenet::FileExists(itn_tagger_path) &&
        wenet::FileExists(itn_verbalizer_path)) {
      LOG(INFO) << "Reading ITN fst" << FLAGS_itn_model_dir;
      post_process_opts.itn = true;
      auto postprocessor =
          std::make_shared<wenet::PostProcessor>(std::move(post_process_opts));
      postprocessor->InitITNResource(itn_tagger_path, itn_verbalizer_path);
      resource->post_processor = postprocessor;
    }
  }

  return resource;
}

}  // namespace wenet

#endif  // DECODER_PARAMS_H_
