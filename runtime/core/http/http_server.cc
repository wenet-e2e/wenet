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

#include "http/http_server.h"

#include <thread>
#include <utility>
#include <vector>

#include "boost/json/src.hpp"
#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;    // from <boost/beast.hpp>
namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

ConnectionHandler::ConnectionHandler(
    tcp::socket&& socket, std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<DecodeResource> decode_resource)
    : socket_(std::move(socket)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)),
      req_(std::make_shared<http::request<http::string_body>>(
          http::verb::post, target_, version_)),
      res_(std::make_shared<http::response<http::string_body>>(http::status::ok,
                                                               version_)) {}

void ConnectionHandler::OnSpeechStart() {
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  decoder_ = std::make_shared<AsrDecoder>(feature_pipeline_, decode_resource_,
                                          *decode_config_);
  // Start decoder thread
  decode_thread_ =
      std::make_shared<std::thread>(&ConnectionHandler::DecodeThreadFunc, this);
}

void ConnectionHandler::OnSpeechEnd() {
  if (feature_pipeline_ != nullptr) {
    feature_pipeline_->set_input_finished();
  }
}

void ConnectionHandler::OnFinalResult(const std::string& result) {
  LOG(INFO) << "Final result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "final_result"}, {"nbest", result}};
  std::string message = json::serialize(rv);
  res_.get()->body() = message;
  http::write(socket_, *res_.get(), ec_);
}

void ConnectionHandler::OnSpeechData(const std::string& message) {
  std::size_t decode_size =
      beast::detail::base64::decoded_size(message.length());
  int num_samples = decode_size / sizeof(int16_t);
  int16_t decode_data[num_samples];  // NOLINT
  beast::detail::base64::decode(decode_data, message.c_str(), message.length());

  // Read binary PCM data
  VLOG(2) << "Received " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  feature_pipeline_->AcceptWaveform(decode_data, num_samples);
}

std::string ConnectionHandler::SerializeResult(bool finish) {
  json::array nbest;
  for (const DecodeResult& path : decoder_->result()) {
    json::object jpath({{"sentence", path.sentence}});
    if (finish) {
      json::array word_pieces;
      for (const WordPiece& word_piece : path.word_pieces) {
        json::object jword_piece({{"word", word_piece.word},
                                  {"start", word_piece.start},
                                  {"end", word_piece.end}});
        word_pieces.emplace_back(jword_piece);
      }
      jpath.emplace("word_pieces", word_pieces);
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == nbest_) {
      break;
    }
  }
  return json::serialize(nbest);
}

void ConnectionHandler::DecodeThreadFunc() {
  try {
    while (true) {
      DecodeState state = decoder_->Decode();
      if (state == DecodeState::kEndFeats || state == DecodeState::kEndpoint) {
        decoder_->Rescoring();
        std::string result = SerializeResult(true);
        OnFinalResult(result);
        break;
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void ConnectionHandler::OnError(const std::string& message) {
  json::value rv = {{"status", "failed"}, {"message", message}};
  res_.get()->body() = json::serialize(rv);
  http::write(socket_, *res_.get(), ec_);
  // Send a TCP shutdown
  socket_.shutdown(tcp::socket::shutdown_send, ec_);
}

void ConnectionHandler::OnText(const std::string& message) {
  LOG(INFO) << message;
  json::value v = json::parse(message);
  if (v.is_object()) {
    json::object obj = v.get_object();
    if (obj.find("nbest") != obj.end()) {
      if (obj["nbest"].is_int64()) {
        nbest_ = obj["nbest"].as_int64();
      } else {
        OnError("integer is expected for nbest option");
      }
    }
  } else {
    OnError("Wrong protocol");
  }
}

void ConnectionHandler::operator()() {
  try {
    http::read(socket_, buffer_, *req_.get(), ec_);
    if (ec_) {
      LOG(ERROR) << ec_;
    } else {
      OnText(req_.get()->base()["config"].to_string());
      OnSpeechStart();
      OnSpeechData(req_.get()->body());
      OnSpeechEnd();
    }
    LOG(INFO) << "Read all pcm data, wait for decoding thread";
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (beast::system_error const& se) {
    LOG(INFO) << se.code().message();
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
  socket_.shutdown(tcp::socket::shutdown_send, ec_);
}

void HttpServer::Start() {
  try {
    auto const address = net::ip::make_address("0.0.0.0");
    tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
    for (;;) {
      // This will receive the new connection
      tcp::socket socket{ioc_};
      // Block until we get a connection
      acceptor.accept(socket);
      // Launch the session, transferring ownership of the socket
      ConnectionHandler handler(std::move(socket), feature_config_,
                                decode_config_, decode_resource_);
      std::thread t(std::move(handler));
      t.detach();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << e.what();
  }
}

}  // namespace wenet
