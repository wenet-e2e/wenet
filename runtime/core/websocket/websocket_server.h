// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

#ifndef WEBSOCKET_WEBSOCKET_SERVER_H_
#define WEBSOCKET_WEBSOCKET_SERVER_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"
#include "glog/logging.h"

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

class ConnectionHandler {
 public:
  ConnectionHandler(tcp::socket&& socket,
                    std::shared_ptr<FeaturePipelineConfig> feature_config,
                    std::shared_ptr<DecodeOptions> decode_config,
                    std::shared_ptr<SymbolTable> symbol_table,
                    std::shared_ptr<TorchAsrModel> model);
  ConnectionHandler(ConnectionHandler&& other);

  void OnSpeechStart();
  void OnSpeechEnd();
  void OnSpeechData(const beast::flat_buffer& buffer);
  void OnError(const std::string& message);
  void OnPartialResult(const std::string& result);
  void OnFinalResult(const std::string& result);
  void DecodeThreadFunc();
  std::string SerializeResult(bool finish);
  void operator()();

 private:
  int nbest_;
  websocket::stream<tcp::socket> ws_;
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<SymbolTable> symbol_table_;
  std::shared_ptr<TorchAsrModel> model_;

  bool got_start_tag_ = false;
  std::shared_ptr<FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<TorchAsrDecoder> decoder_ = nullptr;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;
};

class WebSocketServer {
 public:
  WebSocketServer(int port,
                  std::shared_ptr<FeaturePipelineConfig> feature_config,
                  std::shared_ptr<DecodeOptions> decode_config,
                  std::shared_ptr<SymbolTable> symbol_table,
                  std::shared_ptr<TorchAsrModel> model)
      : port_(port),
        feature_config_(feature_config),
        decode_config_(decode_config),
        symbol_table_(symbol_table),
        model_(model) {}

  void Start();

 private:
  int port_;
  // The io_context is required for all I/O
  asio::io_context ioc_{1};
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<SymbolTable> symbol_table_;
  std::shared_ptr<TorchAsrModel> model_;
  DISALLOW_COPY_AND_ASSIGN(WebSocketServer);
};

}  // namespace wenet

#endif  // WEBSOCKET_WEBSOCKET_SERVER_H_
