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

#ifndef WEBSOCKET_BATCH_CONNECTION_HANDLER_H_
#define WEBSOCKET_BATCH_CONNECTION_HANDLER_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"
#include "boost/json/src.hpp"

#include "decoder/asr_decoder.h"
#include "decoder/batch_asr_decoder.h"
#include "frontend/feature_pipeline.h"
#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

class BatchConnectionHandler {
 public:
  BatchConnectionHandler(
      tcp::socket&& socket,
      std::shared_ptr<FeaturePipelineConfig> feature_config,
      std::shared_ptr<DecodeOptions> decode_config,
      std::shared_ptr<DecodeResource> decode_resource)
    : ws_(std::move(socket)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)) {}

  void operator()() {
    try {
      // Accept the websocket handshake
      ws_.accept();
      for (;;) {
        // This buffer will hold the incoming message
        beast::flat_buffer buffer;
        // Read a message
        ws_.read(buffer);
        if (ws_.got_text()) {
          std::string message = beast::buffers_to_string(buffer.data());
          LOG(INFO) << message;
          OnText(message);
          if (got_end_tag_) {
            break;
          }
        } else {
          if (!got_start_tag_) {
            OnError("Start signal is expected before binary data");
          } else {
            OnSpeechData(buffer);
            break;
          }
        }
      }
      ws_.close(websocket::close_code::normal);
      LOG(INFO) << "ws_ is closed, bye :)";
    } catch (beast::system_error const& se) {
      LOG(INFO) << se.code().message();
      // This indicates that the session was closed
      if (se.code() == websocket::error::closed) {
        OnSpeechEnd();
      }
    } catch (std::exception const& e) {
      LOG(ERROR) << e.what();
      OnError("Decoder got some exception!");
    }
  }

 private:
  void OnSpeechStart() {
    LOG(INFO) << "Received speech start signal, start reading speech";
    got_start_tag_ = true;
    json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
    decoder_ = std::make_shared<BatchAsrDecoder>(
        feature_config_, decode_resource_,
        *decode_config_);
  }

  void OnSpeechEnd() {
    LOG(INFO) << "Received speech end signal";
    got_end_tag_ = true;
  }

  void OnText(const std::string& message) {
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
          if (obj.find("enable_timestamp") != obj.end()) {
            if (obj["enable_timestamp"].is_bool()) {
              enable_timestamp_ = obj["enable_timestamp"].as_bool();
            } else {
              OnError(
                  "boolean true or false is expected for "
                  "enable_timestamp option");
            }
          }
          if (obj.find("batch_lens") != obj.end()) {
            if (obj["batch_lens"].is_array()) {
              batch_lens_.clear();
              auto& batch_lens = obj["batch_lens"].as_array();
              for (size_t i = 0; i < batch_lens.size(); i++) {
                int len = batch_lens[i].as_int64();
                batch_lens_.push_back(len);
              }
            } else {
              OnError("a list of batch_lens should be given");
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

  void OnFinish() {
    // Send finish tag
    json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
  }

  void OnSpeechData(const beast::flat_buffer& buffer) {
    // Read binary PCM data
    std::vector<std::vector<float>> wavs;
    size_t total = std::accumulate(batch_lens_.begin(), batch_lens_.end(), 0);
    VLOG(1) << "buffer size " << buffer.size() << ", batch_lens_ sum " << total;
    CHECK(buffer.size() == total);
    const auto* pcm_data = static_cast<const int16_t*>(buffer.data().data());
    int offset = 0;
    for (int len : batch_lens_) {
      len /= 2;  // lenght of int16_t data
      std::vector<float> wav(len);
      for (size_t i = 0; i < len; i++) {
        wav[i] = static_cast<float>(pcm_data[offset+i]);
      }
      wavs.push_back(std::move(wav));
      offset += len;
    }
    CHECK(decoder_ != nullptr);
    decoder_->Decode(wavs);
    std::string result = decoder_->get_batch_result(nbest_, enable_timestamp_);
    ws_.text(true);
    ws_.write(asio::buffer(result));
  }

  void OnError(const std::string& message) {
    json::value rv = {{"status", "failed"}, {"message", message}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
    // Close websocket
    ws_.close(websocket::close_code::normal);
  }

  int nbest_ = 1;
  bool enable_timestamp_ = false;
  std::vector<int> batch_lens_;
  websocket::stream<tcp::socket> ws_;
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;

  bool got_start_tag_ = false;
  bool got_end_tag_ = false;
  std::shared_ptr<BatchAsrDecoder> decoder_ = nullptr;
};

}  // namespace wenet

#endif  // WEBSOCKET_BATCH_CONNECTION_HANDLER_H_
