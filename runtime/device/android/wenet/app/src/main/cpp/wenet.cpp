#include <jni.h>
#include <string>

#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/symbol_table.h"
#include "decoder/torch_asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"

namespace wenet {

std::shared_ptr<DecodeOptions> decode_config;
std::shared_ptr<FeaturePipelineConfig> feature_config;
std::shared_ptr<FeaturePipeline> feature_pipeline;
std::shared_ptr<SymbolTable> symbol_table;
std::shared_ptr<TorchAsrModel> model;
std::shared_ptr<TorchAsrDecoder> decoder;

jstring init(JNIEnv *env, jobject, jstring jModelPath, jstring jDictPath) {
  model = std::make_shared<TorchAsrModel>();
  const char *pModelPath = (env)->GetStringUTFChars(jModelPath, nullptr);
  std::string modelPath = std::string(pModelPath);
  LOG(INFO) << "model path: " << modelPath;
  model->Read(modelPath);

  const char *pDictPath = (env)->GetStringUTFChars(jDictPath, nullptr);
  std::string dictPath = std::string(pDictPath);
  LOG(INFO) << "dict path: " << dictPath;
  symbol_table = std::make_shared<SymbolTable>(dictPath);

  feature_config = std::make_shared<FeaturePipelineConfig>();
  feature_config->num_bins = 80;
  feature_pipeline = std::make_shared<FeaturePipeline>(*feature_config);

  decode_config = std::make_shared<DecodeOptions>();
  decode_config->chunk_size = 16;

  decoder = std::make_shared<TorchAsrDecoder>(feature_pipeline,
                                              model,
                                              *symbol_table,
                                              *decode_config);

  std::string hello = "Hello from C++";
  return env->NewStringUTF(hello.c_str());
}

void recognize(JNIEnv *env, jobject, jstring jWavPath) {
  const char *pWavPath = (env)->GetStringUTFChars(jWavPath, nullptr);
  std::string wavPath = std::string(pWavPath);
  WavReader wav_reader(wavPath);

  feature_pipeline->AcceptWaveform(std::vector<float>(
      wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  while (true) {
    bool finish = decoder->Decode();
    if (finish) {
      LOG(INFO) << "Final result: " << decoder->result();
      break;
    } else {
      LOG(INFO) << "Partial result: " << decoder->result();
    }
  }
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
      {"init",
       "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
       (void *) wenet::init},
      {"recognize", "(Ljava/lang/String;)V", (void *) wenet::recognize},
  };
  int rc = env->RegisterNatives(
      c, methods, sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
