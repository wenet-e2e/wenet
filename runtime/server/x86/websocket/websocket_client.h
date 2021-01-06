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

#ifndef WEBSOCKET_WEBSOCKET_CLIENT_H_
#define WEBSOCKET_WEBSOCKET_CLIENT_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace net = boost::asio;             // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

class WebSocketClient {
 public:
  WebSocketClient(const std::string& host, int port);

  void AddData(const std::string& data);
  void ReadLoopFunc();
  void Close();

 private:
  void Connect();
  std::string host_;
  int port_;
  net::io_context ioc_;
  websocket::stream<tcp::socket> ws_{ioc_};
  std::unique_ptr<std::thread> t_{nullptr};
};

}  // namespace wenet

#endif  // WEBSOCKET_WEBSOCKET_CLIENT_H_
