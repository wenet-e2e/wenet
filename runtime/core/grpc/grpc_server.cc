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

#include "grpc/grpc_server.h"

namespace wenet {

using grpc::ServerReaderWriter;
using wenet::Request;
using wenet::Response;

GrpcConnectionHandler::GrpcConnectionHandler(
    ServerReaderWriter<Response, Request>* stream,
    std::shared_ptr<Request> request, std::shared_ptr<Response> response,
    std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<DecodeResource> decode_resource)
    : stream_(std::move(stream)),
      request_(std::move(request)),
      response_(std::move(response)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)) {}

void GrpcConnectionHandler::OnSpeechStart() {
  LOG(INFO) << "Received speech start signal, start reading speech";
  got_start_tag_ = true;
  response_->set_status(Response::ok);
  response_->set_type(Response::server_ready);
  stream_->Write(*response_);
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  decoder_ = std::make_shared<AsrDecoder>(feature_pipeline_, decode_resource_,
                                          *decode_config_);
  // Start decoder thread
  decode_thread_ = std::make_shared<std::thread>(
      &GrpcConnectionHandler::DecodeThreadFunc, this);
}

void GrpcConnectionHandler::OnSpeechEnd() {
  LOG(INFO) << "Received speech end signal";
  CHECK(feature_pipeline_ != nullptr);
  feature_pipeline_->set_input_finished();
  got_end_tag_ = true;
}

void GrpcConnectionHandler::OnPartialResult() {
  LOG(INFO) << "Partial result";
  response_->set_status(Response::ok);
  response_->set_type(Response::partial_result);
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnFinalResult() {
  LOG(INFO) << "Final result";
  response_->set_status(Response::ok);
  response_->set_type(Response::final_result);
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnFinish() {
  // Send finish tag
  response_->set_status(Response::ok);
  response_->set_type(Response::speech_end);
  stream_->Write(*response_);
}

void GrpcConnectionHandler::OnSpeechData() {
  // Read binary PCM data
  const int16_t* pcm_data =
      reinterpret_cast<const int16_t*>(request_->audio_data().c_str());
  int num_samples = request_->audio_data().length() / sizeof(int16_t);
  VLOG(2) << "Received " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  feature_pipeline_->AcceptWaveform(pcm_data, num_samples);
}

void GrpcConnectionHandler::SerializeResult(bool finish) {
  for (const DecodeResult& path : decoder_->result()) {
    Response_OneBest* one_best_ = response_->add_nbest();
    one_best_->set_sentence(path.sentence);
    if (finish) {
      for (const WordPiece& word_piece : path.word_pieces) {
        Response_OnePiece* one_piece_ = one_best_->add_wordpieces();
        one_piece_->set_word(word_piece.word);
        one_piece_->set_start(word_piece.start);
        one_piece_->set_end(word_piece.end);
      }
    }
    if (response_->nbest_size() == nbest_) {
      break;
    }
  }
  return;
}

void GrpcConnectionHandler::DecodeThreadFunc() {
  while (true) {
    DecodeState state = decoder_->Decode();
    response_->clear_status();
    response_->clear_type();
    response_->clear_nbest();
    if (state == DecodeState::kEndFeats) {
      decoder_->Rescoring();
      SerializeResult(true);
      OnFinalResult();
      OnFinish();
      stop_recognition_ = true;
      break;
    } else if (state == DecodeState::kEndpoint) {
      decoder_->Rescoring();
      SerializeResult(true);
      OnFinalResult();
      // If it's not continuous decoding, continue to do next recognition
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
        SerializeResult(false);
        OnPartialResult();
      }
    }
  }
}

void GrpcConnectionHandler::operator()() {
  try {
    while (stream_->Read(request_.get())) {
      if (!got_start_tag_) {
        nbest_ = request_->decode_config().nbest_config();
        continuous_decoding_ =
            request_->decode_config().continuous_decoding_config();
        OnSpeechStart();
      } else {
        OnSpeechData();
      }
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
                                decode_config_, decode_resource_);
  std::thread t(std::move(handler));
  t.join();
  return Status::OK;
}
}  // namespace wenet
