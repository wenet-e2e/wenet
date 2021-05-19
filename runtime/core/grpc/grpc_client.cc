// Copyright (c) 2020 Ximalaya Inc (Xiang Lyu)
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

#include "grpc_client.h"

#include "boost/json/src.hpp"

#include "utils/log.h"

namespace wenet {
namespace json = boost::json;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;

GrpcClient::GrpcClient(const std::string& host, int port)
    : host_(host), port_(port) {
  Connect();
  t_.reset(new std::thread(&GrpcClient::ReadLoopFunc, this));
}

void GrpcClient::Connect() {
  channel_ = grpc::CreateChannel(host_ + ":" + std::to_string(port_),
                                 grpc::InsecureChannelCredentials());
  stub_ = ASR::NewStub(channel_);
  context_ = std::make_shared<ClientContext>();
  stream_ = stub_->Recognize(context_.get());
  request_ = std::make_shared<Request>();
  response_ = std::make_shared<Response>();
}

void GrpcClient::SendBinaryData(const void* data, size_t size) {
  request_->set_nbest(nbest_);
  request_->set_continuous_decoding(continuous_decoding_);
  const int16_t* pdata = (int16_t*)data;
  request_->set_audio_data(pdata, size);
  stream_->Write(*request_);
}

void GrpcClient::ReadLoopFunc() {
  try {
    while (stream_->Read(response_.get())) {
      std::string message = (std::string)response_->response_json();
      LOG(INFO) << message;
      json::object obj = json::parse(message).as_object();
      if (obj["status"] != "ok") {
        break;
      }
      if (obj["type"] == "speech_end") {
        done_ = true;
        break;
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void GrpcClient::Join() {
  stream_->WritesDone();
  t_->join();
  Status status = stream_->Finish();
  if (!status.ok()) {
    LOG(INFO) << "Recognize rpc failed.";
  }
}
}  // namespace wenet
