// Copyright (c) 2023 Ximalaya Speech Team (Xiang Lyu)
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

#ifndef HTTP_HTTP_SERVER_H_
#define HTTP_HTTP_SERVER_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>

#include "decoder/asr_decoder.h"
#include "frontend/feature_pipeline.h"
#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;    // from <boost/beast.hpp>
namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

class ConnectionHandler {
 public:
  ConnectionHandler(tcp::socket&& socket,
                    std::shared_ptr<FeaturePipelineConfig> feature_config,
                    std::shared_ptr<DecodeOptions> decode_config,
                    std::shared_ptr<DecodeResource> decode_resource_);
  void operator()();

 private:
  void OnSpeechStart();
  void OnSpeechEnd();
  void OnText(const std::string& message);
  void OnSpeechData(const std::string& message);
  void OnError(const std::string& message);
  void OnFinalResult(const std::string& result);
  void DecodeThreadFunc();
  std::string SerializeResult(bool finish);

  std::string target_ = "/";
  int version_ = 11;
  const bool continuous_decoding_ = false;
  int nbest_ = 1;
  tcp::socket socket_;
  beast::flat_buffer buffer_;
  beast::error_code ec_;
  std::shared_ptr<http::request<http::string_body>> req_;
  std::shared_ptr<http::response<http::string_body>> res_;
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;

  std::shared_ptr<FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<AsrDecoder> decoder_ = nullptr;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;
};

class HttpServer {
 public:
  HttpServer(int port, std::shared_ptr<FeaturePipelineConfig> feature_config,
             std::shared_ptr<DecodeOptions> decode_config,
             std::shared_ptr<DecodeResource> decode_resource)
      : port_(port),
        feature_config_(std::move(feature_config)),
        decode_config_(std::move(decode_config)),
        decode_resource_(std::move(decode_resource)) {}

  void Start();

 private:
  int port_;
  // The io_context is required for all I/O
  net::io_context ioc_{1};
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;
  WENET_DISALLOW_COPY_AND_ASSIGN(HttpServer);
};

}  // namespace wenet

#endif  // HTTP_HTTP_SERVER_H_
