//
//  wenet.cmm
//  WenetDemo

#include "wenet.h"

#include "decoder/asr_decoder.h"
#include "decoder/ios_asr_model.h"
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

- (nullable instancetype)initWithModelPath:(NSString*)modelPath DictPath:(NSString*)dictPath {
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            auto model = std::make_shared<IosAsrModel>();
            model->Read(modelPath.UTF8String);
            resource = std::make_shared<DecodeResource>();
            resource->model = model;
            resource->symbol_table = std::shared_ptr<fst::SymbolTable>(
                                                                       fst::SymbolTable::ReadText(dictPath.UTF8String));
            
            PostProcessOptions post_process_opts;
            resource->post_processor = std::make_shared<PostProcessor>(post_process_opts);
            
            feature_config = std::make_shared<FeaturePipelineConfig>(80, 16000);
            feature_pipeline = std::make_shared<FeaturePipeline>(*feature_config);
            
            decode_config = std::make_shared<DecodeOptions>();
            decode_config->chunk_size = 16;
            decoder = std::make_shared<AsrDecoder>(feature_pipeline, resource, *decode_config);
            
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
    total_result = "";
}

- (void)acceptWaveForm: (float*)pcm: (int)size {
    auto* float_pcm = new float[size];
      for (size_t i = 0; i < size; i++) {
        float_pcm[i] = pcm[i] * 65535;
      }
    feature_pipeline->AcceptWaveform(float_pcm, size);
}

- (void)decode {
    while(true) {
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
            break;
        } else if (state == kEndpoint) {
            LOG(INFO) << "wenet endpoint final result: " << result;
            NSLog(@"wenet endpoint final result: %s", result.c_str());
            total_result += result + "，";
            decoder->ResetContinuousDecoding();
            break;
        } else {
            if (decoder->DecodedSomething()) {
                LOG(INFO) << "wenet partial result: " << result;
            }
        }
    }
}

@end
