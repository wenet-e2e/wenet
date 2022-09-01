// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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


#ifndef DECODER_BATCH_ONNX_ASR_MODEL_H_
#define DECODER_BATCH_ONNX_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "decoder/batch_asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

class BatchOnnxAsrModel : public BatchAsrModel {
 public:
  // Note: Do not call the InitEngineThreads function more than once.
  static void InitEngineThreads(int num_threads = 1);

 public:
  BatchOnnxAsrModel() = default;
  BatchOnnxAsrModel(const BatchOnnxAsrModel& other);
  void Read(const std::string& model_dir, bool is_fp16=false, int gpu_id=0);
  void AttentionRescoring(const std::vector<std::vector<std::vector<int>>>& batch_hyps,
                          const std::vector<std::vector<float>>& ctc_scores,
                          std::vector<std::vector<float>>& attention_scores) override;
  std::shared_ptr<BatchAsrModel> Copy() const override;

  void GetInputOutputInfo(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);
  void ForwardEncoder(
      const batch_feature_t& batch_feats,
      const std::vector<int>& batch_feats_lens,
      std::vector<std::vector<std::vector<float>>>& batch_topk_scores,
      std::vector<std::vector<std::vector<int32_t>>>& batch_topk_indexs) override;

 private:
  int encoder_output_size_ = 0;
  bool is_fp16_ = false;

  // sessions
  // NOTE(Mddct): The Env holds the logging state used by all other objects.
  //  One Env must be created before using any other Onnxruntime functionality.
  static Ort::Env env_;  // shared environment across threads.
  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
  std::shared_ptr<Ort::Session> rescore_session_ = nullptr;

  // node names
  std::vector<const char*> encoder_in_names_, encoder_out_names_;
  std::vector<const char*> rescore_in_names_, rescore_out_names_;

  // cache encoder outs: [encoder_outs, encoder_outs_lens]
  Ort::Value encoder_outs_{nullptr};
  Ort::Value encoder_outs_lens_{nullptr};
};

}  // namespace wenet

#endif  // DECODER_BATCH_ONNX_ASR_MODEL_H_
