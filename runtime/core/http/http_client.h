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

#ifndef HTTP_HTTP_CLIENT_H_
#define HTTP_HTTP_CLIENT_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include "utils/utils.h"

namespace wenet {

namespace beast = boost::beast;  // from <boost/beast.hpp>
namespace http = beast::http;    // from <boost/beast/http.hpp>
namespace net = boost::asio;     // from <boost/asio.hpp>
using tcp = net::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

class HttpClient {
 public:
  HttpClient(const std::string& host, int port);

  void SendBinaryData(const void* data, size_t size);
  void set_nbest(int nbest) { nbest_ = nbest; }

 private:
  void Connect();
  std::string hostname_;
  int port_;
  std::string target_ = "/";
  int version_ = 11;
  int nbest_ = 1;
  const bool continuous_decoding_ = false;
  net::io_context ioc_;
  beast::tcp_stream stream_{ioc_};
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_{http::verb::get, target_, version_};
  http::response<http::string_body> res_{http::status::ok, version_};
  beast::error_code ec_;

  WENET_DISALLOW_COPY_AND_ASSIGN(HttpClient);
};

}  // namespace wenet

#endif  // HTTP_HTTP_CLIENT_H_
