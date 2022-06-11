// Copyright (c) 2021 Mobvoi Inc (authors: Xiaoyu Chen)
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
#include <jni.h>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "post_processor/post_processor.h"
#include "utils/log.h"
#include "utils/string.h"

namespace wenet {

std::shared_ptr<DecodeOptions> decode_config;
std::shared_ptr<FeaturePipelineConfig> feature_config;
std::shared_ptr<FeaturePipeline> feature_pipeline;
std::shared_ptr<AsrDecoder> decoder;
std::shared_ptr<DecodeResource> resource;
DecodeState state = kEndBatch;
std::string total_result;  // NOLINT

void init(JNIEnv* env, jobject, jstring jModelDir) {
  const char* pModelDir = env->GetStringUTFChars(jModelDir, nullptr);

  std::string modelPath = std::string(pModelDir) + "/final.zip";
  std::string dictPath = std::string(pModelDir) + "/units.txt";
  auto model = std::make_shared<TorchAsrModel>();
  model->Read(modelPath);
  LOG(INFO) << "model path: " << modelPath;

  resource = std::make_shared<DecodeResource>();
  resource->model = model;
  resource->symbol_table = std::shared_ptr<fst::SymbolTable>(
          fst::SymbolTable::ReadText(dictPath));
  LOG(INFO) << "dict path: " << dictPath;

  PostProcessOptions post_process_opts;
  resource->post_processor =
    std::make_shared<PostProcessor>(post_process_opts);

  feature_config = std::make_shared<FeaturePipelineConfig>(80, 16000);
  feature_pipeline = std::make_shared<FeaturePipeline>(*feature_config);

  decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = 16;
  decoder = std::make_shared<AsrDecoder>(feature_pipeline, resource,
                                         *decode_config);
}

void reset(JNIEnv *env, jobject) {
  LOG(INFO) << "wenet reset";
  decoder->Reset();
  state = kEndBatch;
  total_result = "";
}

void accept_waveform(JNIEnv *env, jobject, jshortArray jWaveform) {
  jsize size = env->GetArrayLength(jWaveform);
  int16_t* waveform = env->GetShortArrayElements(jWaveform, 0);
  feature_pipeline->AcceptWaveform(waveform, size);
  LOG(INFO) << "wenet accept waveform in ms: " << int(size / 16);
}

void set_input_finished() {
  LOG(INFO) << "wenet input finished";
  feature_pipeline->set_input_finished();
}

void decode_thread_func() {
  while (true) {
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
      total_result += result;
      break;
    } else if (state == kEndpoint) {
      LOG(INFO) << "wenet endpoint final result: " << result;
      total_result += result + "，";
      decoder->ResetContinuousDecoding();
    } else {
      if (decoder->DecodedSomething()) {
        LOG(INFO) << "wenet partial result: " << result;
      }
    }
  }
}

void start_decode() {
  std::thread decode_thread(decode_thread_func);
  decode_thread.detach();
}

jboolean get_finished(JNIEnv *env, jobject) {
  if (state == kEndFeats) {
    LOG(INFO) << "wenet recognize finished";
    return JNI_TRUE;
  }
  return JNI_FALSE;
}

jstring get_result(JNIEnv *env, jobject) {
  std::string result;
  if (decoder->DecodedSomething()) {
    result = decoder->result()[0].sentence;
  }
  LOG(INFO) << "wenet ui result: " << total_result + result;
  return env->NewStringUTF((total_result + result).c_str());
}
}  // namespace wenet

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("com/mobvoi/wenet/Recognize");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
    {"init", "(Ljava/lang/String;)V", reinterpret_cast<void*>(wenet::init)},
    {"reset", "()V", reinterpret_cast<void *>(wenet::reset)},
    {"acceptWaveform", "([S)V",
     reinterpret_cast<void *>(wenet::accept_waveform)},
    {"setInputFinished", "()V",
     reinterpret_cast<void *>(wenet::set_input_finished)},
    {"getFinished", "()Z", reinterpret_cast<void *>(wenet::get_finished)},
    {"startDecode", "()V", reinterpret_cast<void *>(wenet::start_decode)},
    {"getResult", "()Ljava/lang/String;",
     reinterpret_cast<void *>(wenet::get_result)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
