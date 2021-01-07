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

#include "glog/logging.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace net = boost::asio;             // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

void ConnectionHandler(tcp::socket& socket) {  // NOLINT
  try {
    // Construct the stream by moving in the socket
    websocket::stream<tcp::socket> ws{std::move(socket)};
    // Accept the websocket handshake
    ws.accept();
    for (;;) {
      // This buffer will hold the incoming message
      beast::flat_buffer buffer;
      // Read a message
      ws.read(buffer);
      LOG(INFO) << beast::make_printable(buffer.data());
      // Echo the message back
      ws.text(ws.got_text());
      ws.write(buffer.data());
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
    auto const address = net::ip::make_address("0.0.0.0");
    tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
    for (;;) {
      // This will receive the new connection
      tcp::socket socket{ioc_};
      // Block until we get a connection
      acceptor.accept(socket);
      // Launch the session, transferring ownership of the socket
      std::thread{std::bind(ConnectionHandler, std::move(socket))}.detach();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << e.what();
  }
}

}  // namespace wenet
