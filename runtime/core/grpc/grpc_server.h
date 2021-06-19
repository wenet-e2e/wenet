// Copyright (c) 2021 Ximalaya Speech Team (Xiang Lyu)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRPC_GRPC_SERVER_H_
#define GRPC_GRPC_SERVER_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "utils/log.h"

#include "grpc/wenet.grpc.pb.h"

namespace wenet {

using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using wenet::ASR;
using wenet::Request;
using wenet::Response;

class GrpcConnectionHandler {
 public:
  GrpcConnectionHandler(ServerReaderWriter<Response, Request> *stream,
                        std::shared_ptr<Request> request,
                        std::shared_ptr<Response> response,
                        std::shared_ptr<FeaturePipelineConfig> feature_config,
                        std::shared_ptr<DecodeOptions> decode_config,
                        std::shared_ptr<fst::SymbolTable> symbol_table,
                        std::shared_ptr<TorchAsrModel> model,
                        std::shared_ptr<fst::Fst<fst::StdArc>> fst);
  void operator()();

 private:
  void OnSpeechStart();
  void OnSpeechEnd();
  void OnFinish();
  void OnSpeechData();
  void OnPartialResult();
  void OnFinalResult();
  void DecodeThreadFunc();
  void SerializeResult(bool finish);

  bool continuous_decoding_ = false;
  int nbest_ = 1;
  ServerReaderWriter<Response, Request> *stream_;
  std::shared_ptr<Request> request_;
  std::shared_ptr<Response> response_;
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  std::shared_ptr<TorchAsrModel> model_;
  std::shared_ptr<fst::Fst<fst::StdArc>> fst_;

  bool got_start_tag_ = false;
  bool got_end_tag_ = false;
  // When endpoint is detected, stop recognition, and stop receiving data.
  bool stop_recognition_ = false;
  std::shared_ptr<FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<TorchAsrDecoder> decoder_ = nullptr;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;
};

class GrpcServer final : public ASR::Service {
 public:
  GrpcServer(std::shared_ptr<FeaturePipelineConfig> feature_config,
             std::shared_ptr<DecodeOptions> decode_config,
             std::shared_ptr<fst::SymbolTable> symbol_table,
             std::shared_ptr<TorchAsrModel> model,
             std::shared_ptr<fst::Fst<fst::StdArc>> fst)
      : feature_config_(std::move(feature_config)),
        decode_config_(std::move(decode_config)),
        symbol_table_(std::move(symbol_table)),
        model_(std::move(model)),
        fst_(std::move(fst)) {}
  Status Recognize(ServerContext *context,
                   ServerReaderWriter<Response, Request> *reader) override;

 private:
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  std::shared_ptr<TorchAsrModel> model_;
  std::shared_ptr<fst::Fst<fst::StdArc>> fst_;
  DISALLOW_COPY_AND_ASSIGN(GrpcServer);
};

}  // namespace wenet

#endif  // GRPC_GRPC_SERVER_H_
