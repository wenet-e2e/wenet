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


#include <memory>
#include <string>
#include <vector>

#include "decoder/asr_decoder.h"
#include "decoder/batch_asr_decoder.h"
#include "decoder/batch_torch_asr_model.h"
#include "post_processor/post_processor.h"
#include "utils/file.h"
#include "utils/json.h"
#include "utils/string.h"

class BatchRecognizer {
 public:
  explicit BatchRecognizer(const std::string& model_dir, int num_threads=1) {
    // FeaturePipeline init
    feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(80, 16000);
    // Resource init
    resource_ = std::make_shared<wenet::DecodeResource>();
    wenet::BatchTorchAsrModel::InitEngineThreads(num_threads);
    std::string model_path = wenet::JoinPath(model_dir, "final.zip");
    CHECK(wenet::FileExists(model_path));

    auto model = std::make_shared<wenet::BatchTorchAsrModel>();
    model->Read(model_path);
    resource_->batch_model = model;

    // units.txt: E2E model unit
    std::string unit_path = wenet::JoinPath(model_dir, "units.txt");
    CHECK(wenet::FileExists(unit_path));
    resource_->unit_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(unit_path));

    std::string fst_path = wenet::JoinPath(model_dir, "TLG.fst");
    if (wenet::FileExists(fst_path)) {  // With LM
      resource_->fst = std::shared_ptr<fst::Fst<fst::StdArc>>(
          fst::Fst<fst::StdArc>::Read(fst_path));

      std::string symbol_path = wenet::JoinPath(model_dir, "words.txt");
      CHECK(wenet::FileExists(symbol_path));
      resource_->symbol_table = std::shared_ptr<fst::SymbolTable>(
          fst::SymbolTable::ReadText(symbol_path));
    } else {  // Without LM, symbol_table is the same as unit_table
      resource_->symbol_table = resource_->unit_table;
    }

    // Context config init
    context_config_ = std::make_shared<wenet::ContextConfig>();
    decode_options_ = std::make_shared<wenet::DecodeOptions>();
    post_process_opts_ = std::make_shared<wenet::PostProcessOptions>();
  }

  void InitDecoder() {
    CHECK(decoder_ == nullptr);
    // Optional init context graph
    if (context_.size() > 0) {
      context_config_->context_score = context_score_;
      auto context_graph =
          std::make_shared<wenet::ContextGraph>(*context_config_);
      context_graph->BuildContextGraph(context_, resource_->symbol_table);
      resource_->context_graph = context_graph;
    }
    // PostProcessor
    if (language_ == "chs") {  // TODO(Binbin Zhang): CJK(chs, jp, kr)
      post_process_opts_->language_type = wenet::kMandarinEnglish;
    } else {
      post_process_opts_->language_type = wenet::kIndoEuropean;
    }
    resource_->post_processor =
        std::make_shared<wenet::PostProcessor>(*post_process_opts_);
    // Init decoder
    decoder_ = std::make_shared<wenet::BatchAsrDecoder>(feature_config_, resource_,
                                                   *decode_options_);
  }

  std::string Decode(const std::vector<std::string>& wavs) {
    // Init decoder when it is called first time
    if (decoder_ == nullptr) {
      InitDecoder();
    }
    std::vector<std::vector<float>> wavs_float;
    for (auto& wav : wavs) {
      const int16_t* pcm = reinterpret_cast<const int16_t*>(wav.data());
      int pcm_len = wav.size() / sizeof(int16_t);
      std::vector<float> wav_float(pcm_len);
      for (size_t i = 0; i < pcm_len; i++) {
        wav_float[i] = static_cast<float>(*(pcm + i));
      }
      wavs_float.push_back(std::move(wav_float));
    }
    decoder_->Reset();
    decoder_->Decode(wavs_float);
    return UpdateResult();
  }
  
  std::string DecodeData(const std::vector<std::vector<float>>& wavs) {
    // Init decoder when it is called first time
    if (decoder_ == nullptr) {
      InitDecoder();
    }
    decoder_->Reset();
    decoder_->Decode(wavs);
    return UpdateResult();
  }

  std::string UpdateResult() {
    const auto& batch_result = decoder_->batch_result();
    json::JSON obj;
    obj["batch_size"] = batch_result.size();
    obj["batch_result"] = json::Array();
    for (const auto& result : batch_result) {
      json::JSON batch_one;
      batch_one["nbest"] = json::Array();
      for (int i = 0; i < nbest_ && i < result.size(); i++) {
        json::JSON one;
        one["sentence"] = result[i].sentence;
        if (enable_timestamp_) {
          one["word_pieces"] = json::Array();
          for (const auto& word_piece : result[i].word_pieces) {
            json::JSON piece;
            piece["word"] = word_piece.word;
            piece["start"] = word_piece.start;
            piece["end"] = word_piece.end;
            one["word_pieces"].append(piece);
          }
        }
        one["sentence"] = result[i].sentence;
        batch_one["nbest"].append(one);
      }
      obj["batch_result"].append(batch_one);
    }
    return obj.dump();
  }

  void set_nbest(int n) { nbest_ = n; }
  void set_enable_timestamp(bool flag) { enable_timestamp_ = flag; }
  void AddContext(const char* word) { context_.emplace_back(word); }
  void set_context_score(float score) { context_score_ = score; }
  void set_language(const char* lang) { language_ = lang; }

 private:
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::DecodeResource> resource_ = nullptr;
  std::shared_ptr<wenet::DecodeOptions> decode_options_ = nullptr;
  std::shared_ptr<wenet::BatchAsrDecoder> decoder_ = nullptr;
  std::shared_ptr<wenet::ContextConfig> context_config_ = nullptr;
  std::shared_ptr<wenet::PostProcessOptions> post_process_opts_ = nullptr;

  int nbest_ = 1;
  bool enable_timestamp_ = false;
  std::vector<std::string> context_;
  float context_score_;
  std::string language_ = "chs";
};

