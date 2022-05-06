// Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
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

#include "api/wenet_api.h"

#include <memory>
#include <string>
#include <vector>

#include "decoder/asr_decoder.h"
#include "decoder/torch_asr_model.h"


static std::string JoinPath(const std::string& left, const std::string& right) {
  std::string path(left);
  if (path.size() && path.back() != '/') {
      path.push_back('/');
  }
  path.append(right);

  return path;
}


class Recognizer {
 public:
  explicit Recognizer(const std::string& model_dir) {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    // FeaturePipeline init
    feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(80, 16000);
    feature_pipeline_ =
        std::make_shared<wenet::FeaturePipeline>(*feature_config_);
    // Resource init
    resource_ = std::make_shared<wenet::DecodeResource>();
    auto model = std::make_shared<wenet::TorchAsrModel>();
    model->Read(JoinPath(model_dir, "final.zip"));
    resource_->model = model;
    auto symbol_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(JoinPath(model_dir, "words.txt")));
    resource_->symbol_table = symbol_table;
    decode_options_ = std::make_shared<wenet::DecodeOptions>();
    // Decoder init
    decoder_ = std::make_shared<wenet::AsrDecoder>(feature_pipeline_,
        resource_, *decode_options_);
  }

  void Reset() {
    feature_pipeline_->Reset();
    decoder_->Reset();
    result_.clear();
  }

  void Decode(const char* data, int len, int last) {
    using wenet::DecodeState;
    CHECK_EQ(len % 2, 0);
    std::vector<float> wav(len / 2, 0);
    for (int i = 0; i < wav.size(); i++) {
      wav[i] = *reinterpret_cast<const int16_t*>(data + i * 2);
    }
    feature_pipeline_->AcceptWaveform(wav);
    if (last > 0) {
      feature_pipeline_->set_input_finished();
    }

    while (true) {
      // TODO(Binbin Zhang): Process streaming call
      DecodeState state = decoder_->Decode();
      if (state == DecodeState::kEndFeats) {
        decoder_->Rescoring();
        break;
      }
    }
  }

  const char* GetResult() {
    if (decoder_->result().size() > 0) {
      result_ = decoder_->result()[0].sentence;
      return result_.c_str();
    } else {
      return nullptr;
    }
  }

 private:
  // NOTE(Binbin Zhang): All use shared_ptr for clone in the future
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<wenet::DecodeResource> resource_ = nullptr;
  std::shared_ptr<wenet::DecodeOptions> decode_options_ = nullptr;
  std::shared_ptr<wenet::AsrDecoder> decoder_ = nullptr;
  std::string result_;
};


void* wenet_init(const char* model_dir) {
  Recognizer* decoder = new Recognizer(model_dir);
  return reinterpret_cast<void*>(decoder);
}


void wenet_free(void* decoder) {
  delete reinterpret_cast<Recognizer *>(decoder);
}


void wenet_reset(void* decoder) {
  Recognizer *recognizer = reinterpret_cast<Recognizer *>(decoder);
  recognizer->Reset();
}


void wenet_decode(void* decoder,
                  const char* data,
                  int len,
                  int last) {
  Recognizer *recognizer = reinterpret_cast<Recognizer *>(decoder);
  recognizer->Decode(data, len, last);
}


const char* wenet_get_result(void* decoder) {
  Recognizer *recognizer = reinterpret_cast<Recognizer *>(decoder);
  return recognizer->GetResult();
}


void wenet_set_log_level(int level) {
  FLAGS_logtostderr = true;
  FLAGS_v = level;
}
