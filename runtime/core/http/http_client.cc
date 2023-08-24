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

namespace beast = boost::beast;  // from <boost/beast.hpp>
namespace http = beast::http;    // from <boost/beast/http.hpp>
namespace net = boost::asio;     // from <boost/asio.hpp>
using tcp = net::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

HttpClient::HttpClient(const std::string& hostname, int port)
    : hostname_(hostname), port_(port) {
  Connect();
}

void HttpClient::Connect() {
  tcp::resolver resolver{ioc_};
  // Look up the domain name
  auto const results = resolver.resolve(hostname_, std::to_string(port_));
  stream_.connect(results);
}

void HttpClient::SendBinaryData(const void* data, size_t size) {
  try {
    json::value start_tag = {{"nbest", nbest_},
                             {"continuous_decoding", continuous_decoding_}};
    std::string config = json::serialize(start_tag);
    req_.set("config", config);
    std::size_t encode_size = beast::detail::base64::encoded_size(size);
    char encode_data[encode_size];  // NOLINT
    beast::detail::base64::encode(encode_data, data, size);
    req_.body() = encode_data;
    req_.prepare_payload();
    http::write(stream_, req_, ec_);

    http::read(stream_, buffer_, res_);
    std::string message = res_.body();
    json::object obj = json::parse(message).as_object();
    LOG(INFO) << message;
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
  stream_.socket().shutdown(tcp::socket::shutdown_both, ec_);
}

}  // namespace wenet
