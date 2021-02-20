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

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "websocket/websocket_server.h"

DEFINE_int32(port, 10086, "websocket listening port");
DEFINE_int32(num_bins, 80, "num mel bins for fbank feature");
DEFINE_int32(chunk_size, 16, "decoding chunk size");
DEFINE_int32(num_left_chunks, -1, "left chunks in decoding");
DEFINE_string(model_path, "", "pytorch exported model path");
DEFINE_string(wav_path, "", "wav path");
DEFINE_string(dict_path, "", "dict path");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto feature_config = std::make_shared<wenet::FeaturePipelineConfig>();
  feature_config->num_bins = FLAGS_num_bins;
  auto decode_config = std::make_shared<wenet::DecodeOptions>();
  decode_config->chunk_size = FLAGS_chunk_size;
  decode_config->num_left_chunks = FLAGS_num_left_chunks;
  auto symbol_table = std::make_shared<wenet::SymbolTable>(FLAGS_dict_path);
  auto model = std::make_shared<wenet::TorchAsrModel>();
  model->Read(FLAGS_model_path);

  wenet::WebSocketServer server(FLAGS_port, feature_config, decode_config,
                                symbol_table, model);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}
