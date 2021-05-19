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

#include "boost/json/src.hpp"
#include "grpc_server.h"

namespace wenet {

namespace json = boost::json;
using grpc::ServerReaderWriter;
using wenet::Request;
using wenet::Response;

GrpcConnectionHandler::GrpcConnectionHandler(
    ServerReaderWriter<Response, Request>* stream,
    std::shared_ptr<Request> request,
    std::shared_ptr<Response> response,
    std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<fst::SymbolTable> symbol_table,
    std::shared_ptr<TorchAsrModel> model,
    std::shared_ptr<fst::StdVectorFst> fst)
    : stream_(std::move(stream)),
      request_(std::move(request)),
      response_(std::move(response)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      symbol_table_(std::move(symbol_table)),
      model_(std::move(model)),
      fst_(std::move(fst)) {
}

void GrpcConnectionHandler::OnSpeechStart() {
  LOG(INFO) << "Recieved speech start signal, start reading speech";
  got_start_tag_ = true;
  json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
  response_->set_response_json(json::serialize(rv));
  stream_->Write(*response_);
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  decoder_ = std::make_shared<TorchAsrDecoder>(
      feature_pipeline_, model_, symbol_table_, *decode_config_, fst_);
  // Start decoder thread
  decode_thread_ = std::make_shared<std::thread>(
      &GrpcConnectionHandler::DecodeThreadFunc, this);
}

void GrpcConnectionHandler::OnSpeechEnd() {
  LOG(INFO) << "Recieved speech end signal";
  CHECK(feature_pipeline_ != nullptr);
  feature_pipeline_->set_input_finished();
  got_end_tag_ = true;
}

void GrpcConnectionHandler::OnPartialResult(const std::string& result) {
  LOG(INFO) << "Partial result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "partial_result"}, {"nbest", result}};
  response_->set_response_json(json::serialize(rv));
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnFinalResult(const std::string& result) {
  LOG(INFO) << "Final result: " << result;
  json::value rv = {
      {"status", "ok"}, {"type", "final_result"}, {"nbest", result}};
  response_->set_response_json(json::serialize(rv));
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnFinish() {
  // Send finish tag
  json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
  response_->set_response_json(json::serialize(rv));
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnSpeechData() {
  // Read binary PCM data
  const int16_t* pdata = (int16_t*)request_->audio_data().c_str();
  int num_samples = request_->audio_data().length() / sizeof(int16_t);
  std::vector<float> pcm_data(num_samples);
  for (int i = 0; i < num_samples; i++) {
    pcm_data[i] = static_cast<float>(*pdata);
    pdata++;
  }
  VLOG(2) << "Recieved " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  feature_pipeline_->AcceptWaveform(pcm_data);
}

std::string GrpcConnectionHandler::SerializeResult(bool finish) {
  json::array nbest;
  for (const DecodeResult& path : decoder_->result()) {
    json::object jpath({{"sentence", path.sentence}});
    if (finish) {
      json::array word_pieces;
      for (const WordPiece& word_piece : path.word_pieces) {
        json::object jword_piece({{"word", word_piece.word},
                                  {"start", word_piece.start},
                                  {"end", word_piece.end}});
        word_pieces.emplace_back(jword_piece);
      }
      jpath.emplace("word_pieces", word_pieces);
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == nbest_) {
      break;
    }
  }
  return json::serialize(nbest);
}

void GrpcConnectionHandler::DecodeThreadFunc() {
  while (true) {
    DecodeState state = decoder_->Decode();
    if (state == DecodeState::kEndFeats) {
      decoder_->Rescoring();
      std::string result = SerializeResult(true);
      OnFinalResult(result);
      OnFinish();
      stop_recognition_ = true;
      break;
    } else if (state == DecodeState::kEndpoint) {
      decoder_->Rescoring();
      std::string result = SerializeResult(true);
      OnFinalResult(result);
      // If it's not continuous decoidng, continue to do next recognition
      // otherwise stop the recognition
      if (continuous_decoding_) {
        decoder_->ResetContinuousDecoding();
      } else {
        OnFinish();
        stop_recognition_ = true;
        break;
      }
    } else {
      if (decoder_->DecodedSomething()) {
        std::string result = SerializeResult(false);
        OnPartialResult(result);
      }
    }
  }
}

void GrpcConnectionHandler::operator()() {
  try {
    while (stream_->Read(request_.get())) {
      if (!got_start_tag_) {
        nbest_ = (int)request_->nbest();
        continuous_decoding_ = (bool)request_->continuous_decoding();
        OnSpeechStart();
      }
      OnSpeechData();
    }
    OnSpeechEnd();
    LOG(INFO) << "Read all pcm data, wait for decoding thread";
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

Status GrpcServer::Recognize(ServerContext* context,
                             ServerReaderWriter<Response, Request>* stream) {
  LOG(INFO) << "Get Recognize request" << std::endl;
  auto request = std::make_shared<Request>();
  auto response = std::make_shared<Response>();
  GrpcConnectionHandler handler(stream, request, response, feature_config_,
                                decode_config_, symbol_table_, model_, fst_);
  std::thread t(std::move(handler));
  t.join();
  return Status::OK;
}
}  // namespace wenet
