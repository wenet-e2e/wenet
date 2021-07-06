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

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "decoder/params.h"
#include "grpc/grpc_server.h"
#include "utils/log.h"

DEFINE_int32(port, 10086, "grpc listening port");
DEFINE_int32(workers, 4, "grpc num workers");

using grpc::Server;
using grpc::ServerBuilder;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto model = wenet::InitTorchAsrModelFromFlags();
  auto symbol_table = wenet::InitSymbolTableFromFlags();
  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto fst = wenet::InitFstFromFlags();

  wenet::GrpcServer service(feature_config, decode_config, symbol_table, model,
                            fst);
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  std::string address("0.0.0.0:" + std::to_string(FLAGS_port));
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS,
                              FLAGS_workers);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server->Wait();
  google::ShutdownGoogleLogging();
  return 0;
}
