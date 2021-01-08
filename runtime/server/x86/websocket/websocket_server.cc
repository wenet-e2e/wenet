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

#include "websocket/websocket_server.h"

#include <thread>
#include <utility>
#include <vector>

#include "boost/json/src.hpp"
#include "glog/logging.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

void ConnectionHandler::operator()() {
  try {
    // Accept the websocket handshake
    ws_.accept();
    bool got_start_tag = false;
    for (;;) {
      // This buffer will hold the incoming message
      beast::flat_buffer buffer;
      // Read a message
      ws_.read(buffer);
      if (ws_.got_text()) {
        std::string message = beast::buffers_to_string(buffer.data());
        LOG(INFO) << message;
        json::value v = json::parse(message);
        if (v.is_object()) {
          json::object obj = v.get_object();
          if (obj.find("signal") != obj.end()) {
            json::string signal = obj["signal"].as_string();
            if (signal == "start") {
              LOG(INFO) << "Start recieve data";
              got_start_tag = true;
              json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
              ws_.text(true);
              ws_.write(asio::buffer(json::serialize(rv)));
            } else if (signal == "end") {
              json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
              ws_.text(true);
              ws_.write(asio::buffer(json::serialize(rv)));
              LOG(INFO) << "Stop recieve data";
            } else {
              // TODO(Binbin Zhang): error handle
            }
          } else {
            // TODO(Binbin Zhang): error handle
          }
        }
      } else {
        if (!got_start_tag) {
          // TODO(Binbin Zhang): error handle
        } else {
          // Read binary PCM data
          int num_samples = buffer.size() / sizeof(int16_t);
          std::vector<float> pcm_data(num_samples);
          const int16_t* pdata = static_cast<int16_t*>(buffer.data().data());
          for (int i = 0; i < num_samples; i++) {
            pcm_data[i] = static_cast<float>(*pdata);
            pdata++;
          }
          LOG(INFO) << "Recieved " << buffer.size() << " " << pcm_data[0];
        }
      }
    }
  } catch (beast::system_error const& se) {
    // This indicates that the session was closed
    if (se.code() != websocket::error::closed) {
      LOG(ERROR) << se.code().message();
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void WebSocketServer::Start() {
  try {
    auto const address = asio::ip::make_address("0.0.0.0");
    tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
    for (;;) {
      // This will receive the new connection
      tcp::socket socket{ioc_};
      // Block until we get a connection
      acceptor.accept(socket);
      // Launch the session, transferring ownership of the socket
      ConnectionHandler handler(std::move(socket), feature_config_,
                                decode_config_, symbol_table_, model_);
      std::thread t(std::move(handler));
      t.detach();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << e.what();
  }
}

}  // namespace wenet
