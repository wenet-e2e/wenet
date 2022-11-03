// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Han Qi (qihan@baidu.com, Kunlunxin Inc)
//                    Hehe Pan (panhehe@baidu.com, Kunlunxin Inc)
//                    Zikui Yan (yanzikui@baidu.com, Kunlunxin Inc)
// All Rights Reserved.
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

#ifndef RUNTIME_KUNLUN_XPU_XPU_ASR_MODEL_H_
#define RUNTIME_KUNLUN_XPU_XPU_ASR_MODEL_H_

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

#include "xpu_conformer.h"  // NOLINT

namespace wenet {

class XPUAsrModel : public AsrModel {
  typedef float16 T;
  typedef int16_t TW;

 public:
  // Note: Do not call the InitEngineThreads function more than once.
  void SetEngineThreads(int num_threads = 1);

 public:
  XPUAsrModel() = default;
  XPUAsrModel(const XPUAsrModel& other);
  void SetDeviceId(int dev_id);
  void Read(const std::string& model_dir);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

 private:
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;

  // XPU device id
  int device_id_ = 0;
  int real_threads_number = 1;

  // XPU Conformer EncoderParam and DecoderParam
  ConformerEncoderParam<T, TW> encoder_param;
  ConformerDecoderParam<T, TW> decoder_param;

  // XPU input and weights params
  using INPUT_LENGTH_CPU_TUPLE = std::tuple<std::vector<int>, std::vector<int>>;
  using INPUT_XPU_INFO_TUPLE = std::tuple<float*, std::vector<int>>;
  INPUT_LENGTH_CPU_TUPLE input_lenghts_cpu_info;
  INPUT_XPU_INFO_TUPLE input_xpu_info;
  INPUT_XPU_INFO_TUPLE xpu_mask_info_float;

  // XPU encoder and decoder outputs
  T* encoder_out = nullptr;
  T* ctc_probs = nullptr;

  // XPU runtime params
  void* l3ptr = nullptr;
  XPUStream stream;
  std::shared_ptr<api::Context> ctx_xpu_ptr;
  std::shared_ptr<api::ctx_guard> RAII_GUARD;

  int batch, max_seqlen, q_seqlen;

  // caches
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
};

}  // namespace wenet

#endif  // RUNTIME_KUNLUN_XPU_XPU_ASR_MODEL_H_
