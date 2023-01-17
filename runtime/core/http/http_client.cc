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

#include "http/http_client.h"

#include "boost/json/src.hpp"

#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;     // from <boost/beast.hpp>
namespace http = beast::http;       // from <boost/beast/http.hpp>
namespace net = boost::asio;        // from <boost/asio.hpp>
using tcp = net::ip::tcp;           // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

HttpClient::HttpClient(const std::string& hostname, int port)
    : hostname_(hostname), port_(port) {
  Connect();
  t_.reset(new std::thread(&HttpClient::ReadLoopFunc, this));
}

void HttpClient::Connect() {
  tcp::resolver resolver{ioc_};
  // Look up the domain name
  auto const results = resolver.resolve(hostname_, std::to_string(port_));
  stream_.connect(results);
}

void HttpClient::SendBinaryData(const void* data, size_t size) {
  req_.body().data = const_cast<void*>(data);
  req_.body().size = size;
  req_.body().more = true;
  http::write(stream_, sr_, ec_);
}

void HttpClient::Close() {
  beast::error_code ec;
  stream_.socket().shutdown(tcp::socket::shutdown_both, ec);
}

void HttpClient::ReadLoopFunc() {
  beast::error_code ec;
  beast::flat_buffer buffer;
  http::parser<false, http::buffer_body> p;
  char arr[1024];
  std::string message = "";
  try {
    http::read_header(stream_, buffer, p, ec);
    while (!p.is_done()) {
      p.get().body().data = arr;
      p.get().body().size = sizeof(arr);
      http::read_some(stream_, buffer, p, ec);
      if (ec && ec != http::error::need_buffer) {
        LOG(ERROR) << ec;
        break;
      }
      message += std::string(arr).substr(0, sizeof(arr) - p.get().body().size);
      try {
        json::object obj = json::parse(message).as_object();
        LOG(INFO) << message;
        message = "";
        if (obj["type"] == "speech_end") {
          done_ = true;
          break;
        }
      }
      catch (json::system_error const& e) {
        continue;
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
  stream_.socket().shutdown(tcp::socket::shutdown_both, ec);
}

void HttpClient::Join() { t_->join(); }

void HttpClient::SendStartSignal() {
  json::value start_tag = {{"signal", "start"},
                           {"nbest", nbest_},
                           {"continuous_decoding", continuous_decoding_}};
  std::string start_message = json::serialize(start_tag);
  req_.set(http::field::transfer_encoding, "chunked");
  req_.set("config", start_message);
  http::write_header(stream_, sr_, ec_);
}

void HttpClient::SendEndSignal() {
  req_.body().data = nullptr;
  req_.body().more = false;
  http::write(stream_, sr_, ec_);
}

}  // namespace wenet
