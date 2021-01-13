#include <jni.h>
#include <string>

#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"

jstring test(JNIEnv *env, jobject) {
  torch::Tensor tensor = torch::eye(3);
  LOG(INFO) << tensor;
  std::string hello = "Hello from C++";
  return env->NewStringUTF(hello.c_str());
}

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
      {"test", "()Ljava/lang/String;", (void *) test},
  };
  int rc = env->RegisterNatives(c, methods, sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
