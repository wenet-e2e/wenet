// Copyright (c) 2022 Dan Ma (1067837450@qq.com)
//
//  wenet.mm
//  WenetDemo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "wenet.h"

#define IOS

#include "decoder/asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "post_processor/post_processor.h"
#include "utils/log.h"
#include "utils/string.h"

using namespace wenet;

@implementation Wenet {
@protected
  std::shared_ptr<DecodeOptions> decode_config;
  std::shared_ptr<FeaturePipelineConfig> feature_config;
  std::shared_ptr<FeaturePipeline> feature_pipeline;
  std::shared_ptr<AsrDecoder> decoder;
  std::shared_ptr<DecodeResource> resource;
  DecodeState state;
  std::string total_result;
}

- (nullable instancetype)initWithModelPath:
(NSString*)modelPath DictPath:(NSString*)dictPath {
  self = [super init];
  if (self) {
    try {
      auto qengines = at::globalContext().supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK)
          != qengines.end()) {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
      }
      auto model = std::make_shared<TorchAsrModel>();
      model->Read(modelPath.UTF8String);
      resource = std::make_shared<DecodeResource>();
      resource->model = model;
      resource->symbol_table = std::shared_ptr<fst::SymbolTable>
      (fst::SymbolTable::ReadText(dictPath.UTF8String));

      PostProcessOptions post_process_opts;
      resource->post_processor =
      std::make_shared<PostProcessor>(post_process_opts);

      feature_config = std::make_shared<FeaturePipelineConfig>(80, 16000);
      feature_pipeline = std::make_shared<FeaturePipeline>(*feature_config);

      decode_config = std::make_shared<DecodeOptions>();
      decode_config->chunk_size = 16;
      decoder = std::make_shared<AsrDecoder>(feature_pipeline,
                                             resource,
                                             *decode_config);

      state = kEndBatch;
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }

  return self;
}

- (void)reset {
  decoder->Reset();
  state = kEndBatch;
  total_result.clear();
}

- (void)acceptWaveForm: (float*)pcm: (int)size {
  auto* float_pcm = new float[size];
  for (size_t i = 0; i < size; i++) {
    float_pcm[i] = pcm[i] * 65535;
  }
  feature_pipeline->AcceptWaveform(float_pcm, size);
}

- (void)decode {
  state = decoder->Decode();
  if (state == kEndFeats || state == kEndpoint) {
    decoder->Rescoring();
  }

  std::string result;
  if (decoder->DecodedSomething()) {
    result = decoder->result()[0].sentence;
  }

  if (state == kEndFeats) {
    LOG(INFO) << "wenet endfeats final result: " << result;
    NSLog(@"wenet endfeats final result: %s", result.c_str());
    total_result += result;
  } else if (state == kEndpoint) {
    LOG(INFO) << "wenet endpoint final result: " << result;
    NSLog(@"wenet endpoint final result: %s", result.c_str());
    total_result += result + "，";
    decoder->ResetContinuousDecoding();
  } else {
    if (decoder->DecodedSomething()) {
      LOG(INFO) << "wenet partial result: " << result;
      NSLog(@"wenet partial result: %s", result.c_str());
    }
  }
}

- (NSString *)get_result {
  std::string result;
  if (decoder->DecodedSomething()) {
    result = decoder->result()[0].sentence;
  }
  std::string final_result = total_result + result;
  LOG(INFO) << "wenet ui result: " << final_result;
  NSLog(@"wenet ui result: %s", final_result.c_str());
  return [NSString stringWithUTF8String:final_result.c_str()];
}

@end
