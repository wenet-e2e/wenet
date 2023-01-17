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

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

ConnectionHandler::ConnectionHandler(
    tcp::socket&& socket, std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<DecodeResource> decode_resource)
    : socket_(std::move(socket)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)),
      res_(std::make_shared<http::response<http::buffer_body>>(http::status::ok,
                                                               version_)),
      sr_(std::make_shared<http::response_serializer<http::buffer_body>>(
          *res_.get())) {}

void ConnectionHandler::OnSpeechStart() {
  LOG(INFO) << "Received speech start signal, start reading speech";
  got_start_tag_ = true;
  res_.get()->set(http::field::transfer_encoding, "chunked");
  http::write_header(socket_, *sr_.get(), ec_);

  json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
  std::string message = json::serialize(rv);
  res_.get()->body().data = const_cast<char*>(message.c_str());
  res_.get()->body().size = message.length();
  res_.get()->body().more = true;
  http::write(socket_, *sr_.get(), ec_);
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  decoder_ = std::make_shared<AsrDecoder>(feature_pipeline_, decode_resource_,
                                          *decode_config_);
  // Start decoder thread
  decode_thread_ =
      std::make_shared<std::thread>(&ConnectionHandler::DecodeThreadFunc, this);
}

void ConnectionHandler::OnSpeechEnd() {
  LOG(INFO) << "Received speech end signal";
  if (feature_pipeline_ != nullptr) {
    feature_pipeline_->set_input_finished();
  }
  got_end_tag_ = true;
}

void ConnectionHandler::OnPartialResult(const std::string& result) {
  LOG(INFO) << "Partial result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "partial_result"}, {"nbest", result}};
  std::string message = json::serialize(rv);
  res_.get()->body().data = const_cast<char*>(message.c_str());
  res_.get()->body().size = message.length();
  res_.get()->body().more = true;
  http::write(socket_, *sr_.get(), ec_);
}

void ConnectionHandler::OnFinalResult(const std::string& result) {
  LOG(INFO) << "Final result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "final_result"}, {"nbest", result}};
  std::string message = json::serialize(rv);
  res_.get()->body().data = const_cast<char*>(message.c_str());
  res_.get()->body().size = message.length();
  res_.get()->body().more = true;
  http::write(socket_, *sr_.get(), ec_);
}

void ConnectionHandler::OnFinish() {
  // Send finish tag
  json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
  std::string message = json::serialize(rv);
  res_.get()->body().data = const_cast<char*>(message.c_str());
  res_.get()->body().size = message.length();
  res_.get()->body().more = false;
  http::write(socket_, *sr_.get(), ec_);
}

void ConnectionHandler::OnSpeechData(const int16_t* pcm_data,
                                     size_t num_samples) {
  // Read binary PCM data
  VLOG(2) << "Received " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  feature_pipeline_->AcceptWaveform(pcm_data, num_samples);
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
      if (state == DecodeState::kEndFeats) {
        decoder_->Rescoring();
        std::string result = SerializeResult(true);
        OnFinalResult(result);
        OnFinish();
        stop_recognition_ = true;
        break;
      } else if (state == DecodeState::kEndpoint) {
        decoder_->Rescoring();
        std::string result = SerializeResult(true);
        OnFinalResult(result);
        // If it's not continuous decoding, continue to do next recognition
        // otherwise stop the recognition
        if (continuous_decoding_) {
          decoder_->ResetContinuousDecoding();
        } else {
          OnFinish();
          stop_recognition_ = true;
          break;
        }
      } else {
        if (decoder_->DecodedSomething()) {
          std::string result = SerializeResult(false);
          OnPartialResult(result);
        }
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void ConnectionHandler::OnError(const std::string& message) {
  json::value rv = {{"status", "failed"}, {"message", message}};
  res_.get()->body().data = const_cast<char*>(message.data());
  res_.get()->body().size = sizeof(message);
  res_.get()->body().more = false;
  http::write(socket_, *sr_.get(), ec_);
  // Send a TCP shutdown
  socket_.shutdown(tcp::socket::shutdown_send, ec_);
}

void ConnectionHandler::OnText(const std::string& message) {
  LOG(INFO) << message;
  json::value v = json::parse(message);
  if (v.is_object()) {
    json::object obj = v.get_object();
    if (obj.find("signal") != obj.end()) {
      json::string signal = obj["signal"].as_string();
      if (signal == "start") {
        if (obj.find("nbest") != obj.end()) {
          if (obj["nbest"].is_int64()) {
            nbest_ = obj["nbest"].as_int64();
          } else {
            OnError("integer is expected for nbest option");
          }
        }
        if (obj.find("continuous_decoding") != obj.end()) {
          if (obj["continuous_decoding"].is_bool()) {
            continuous_decoding_ = obj["continuous_decoding"].as_bool();
          } else {
            OnError(
                "boolean true or false is expected for "
                "continuous_decoding option");
          }
        }
        OnSpeechStart();
      } else if (signal == "end") {
        OnSpeechEnd();
      } else {
        OnError("Unexpected signal type");
      }
    } else {
      OnError("Wrong message header");
    }
  } else {
    OnError("Wrong protocol");
  }
}

void ConnectionHandler::operator()() {
  beast::error_code ec;
  beast::flat_buffer buffer;
  http::parser<true, http::buffer_body> p;
  int16_t arr[8000];
  try {
    http::read_header(socket_, buffer, p, ec);
    OnText(p.get().base()["config"].to_string());
    while (!p.is_done()) {
      p.get().body().data = arr;
      p.get().body().size = sizeof(arr);
      http::read(socket_, buffer, p, ec);
      if (ec && ec != http::error::need_buffer) {
        LOG(ERROR) << ec;
        break;
      }
      if (!got_start_tag_) {
        OnError("Start signal is expected before binary data");
        break;
      } else {
        if (stop_recognition_) {
          break;
        }
        OnSpeechData(arr,
                     (sizeof(arr) - p.get().body().size) / sizeof(int16_t));
      }
    }
    OnText("{\"signal\":\"end\"}");
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
  socket_.shutdown(tcp::socket::shutdown_send, ec);
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
