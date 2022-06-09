// Copyright (c) 2021 Ximalaya Speech Team (Xiang Lyu)
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

#ifndef GRPC_GRPC_CLIENT_H_
#define GRPC_GRPC_CLIENT_H_

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "grpc/wenet.grpc.pb.h"
#include "utils/utils.h"

namespace wenet {

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using wenet::ASR;
using wenet::Request;
using wenet::Response;

class GrpcClient {
 public:
  GrpcClient(const std::string& host, int port, int nbest,
             bool continuous_decoding);

  void SendBinaryData(const void* data, size_t size);
  void ReadLoopFunc();
  void Join();
  bool done() const { return done_; }

 private:
  void Connect();
  std::string host_;
  int port_;
  std::shared_ptr<Channel> channel_{nullptr};
  std::unique_ptr<ASR::Stub> stub_{nullptr};
  std::shared_ptr<ClientContext> context_{nullptr};
  std::unique_ptr<ClientReaderWriter<Request, Response>> stream_{nullptr};
  std::shared_ptr<Request> request_{nullptr};
  std::shared_ptr<Response> response_{nullptr};
  int nbest_ = 1;
  bool continuous_decoding_ = false;
  bool done_ = false;
  std::unique_ptr<std::thread> t_{nullptr};

  WENET_DISALLOW_COPY_AND_ASSIGN(GrpcClient);
};

}  // namespace wenet

#endif  // GRPC_GRPC_CLIENT_H_
