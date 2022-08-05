// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
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


#ifndef DECODER_BATCH_ASR_DECODER_H_
#define DECODER_BATCH_ASR_DECODER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "fst/symbol-table.h"

#include "decoder/batch_asr_model.h"
#include "decoder/asr_decoder.h"
#include "decoder/context_graph.h"
#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/ctc_wfst_beam_search.h"
#include "decoder/search_interface.h"
#include "frontend/feature_pipeline.h"
#include "post_processor/post_processor.h"
#include "utils/utils.h"
#include "frontend/fbank.h"

namespace wenet {

// Torch ASR batch decoder
class BatchAsrDecoder {
 public:
  BatchAsrDecoder(std::shared_ptr<FeaturePipelineConfig> feature_config,
             std::shared_ptr<DecodeResource> resource,
             const DecodeOptions& opts);
  void Decode(const std::vector<std::vector<float>>& wavs);
  void Reset();

  int frame_shift_in_ms() const {
    return model_->subsampling_rate() *
           feature_config_->frame_shift * 1000 /
           feature_config_->sample_rate;
  }
  int feature_frame_shift_in_ms() const {
    return feature_config_->frame_shift * 1000 /
           feature_config_->sample_rate;
  }
  const std::vector<DecodeResult>& result() const { return result_; }
  const std::vector<std::vector<DecodeResult>>& batch_result() const { return batch_result_; }

 private:
  Fbank fbank_;
  void AttentionRescoring(int batch_index);

  void UpdateResult(bool finish = false);

  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<BatchAsrModel> model_;
  std::shared_ptr<PostProcessor> post_processor_;

  std::shared_ptr<fst::Fst<fst::StdArc>> fst_ = nullptr;
  // output symbol table
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  // e2e unit symbol table
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
  const DecodeOptions& opts_;
  int global_frame_offset_ = 0;
  const int time_stamp_gap_ = 100;  // timestamp gap between words in a sentence

  std::unique_ptr<SearchInterface> searcher_;

  std::vector<DecodeResult> result_;
  std::vector<std::vector<DecodeResult>> batch_result_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(BatchAsrDecoder);
};

}  // namespace wenet

#endif  // DECODER_BATCH_ASR_DECODER_H_
