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

#include "decoder/params.h"
#include "utils/log.h"
#include "websocket/websocket_server.h"

DEFINE_int32(port, 10086, "websocket listening port");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto model = wenet::InitTorchAsrModelFromFlags();
  auto symbol_table = wenet::InitSymbolTableFromFlags();
  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto fst = wenet::InitFstFromFlags();

  wenet::WebSocketServer server(FLAGS_port, feature_config, decode_config,
                                symbol_table, model, fst);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}
