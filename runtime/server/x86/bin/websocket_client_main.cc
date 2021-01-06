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

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "websocket/websocket_client.h"

DEFINE_string(host, "127.0.0.1", "host of websocket server");
DEFINE_int32(port, 10086, "port of websocket server");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  wenet::WebSocketClient client(FLAGS_host, FLAGS_port);
  std::string message;

  while (true) {
    std::cout << ">>";
    std::cin >> message;
    client.AddData(message);
  }
  return 0;
}
