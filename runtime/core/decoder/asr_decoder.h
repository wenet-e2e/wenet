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


#ifndef DECODER_ASR_DECODER_H_
#define DECODER_ASR_DECODER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "fst/symbol-table.h"

#include "decoder/asr_model.h"
#include "decoder/context_graph.h"
#include "decoder/ctc_endpoint.h"
#include "decoder/ctc_prefix_beam_search.h"
#include "decoder/ctc_wfst_beam_search.h"
#include "decoder/search_interface.h"
#include "frontend/feature_pipeline.h"
#include "post_processor/post_processor.h"
#include "utils/utils.h"

namespace wenet {

struct DecodeOptions {
  // chunk_size is the frame number of one chunk after subsampling.
  // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
  // one chunk are 64 = 16*4
  int chunk_size = 16;
  int num_left_chunks = -1;

  // final_score = rescoring_weight * rescoring_score + ctc_weight * ctc_score;
  // rescoring_score = left_to_right_score * (1 - reverse_weight) +
  // right_to_left_score * reverse_weight
  // Please note the concept of ctc_scores in the following two search
  // methods are different.
  // For CtcPrefixBeamSearch, it's a sum(prefix) score + context score
  // For CtcWfstBeamSearch, it's a max(viterbi) path score + context score
  // So we should carefully set ctc_weight according to the search methods.
  float ctc_weight = 0.5;
  float rescoring_weight = 1.0;
  float reverse_weight = 0.0;
  CtcEndpointConfig ctc_endpoint_config;
  CtcPrefixBeamSearchOptions ctc_prefix_search_opts;
  CtcWfstBeamSearchOptions ctc_wfst_search_opts;
};

struct WordPiece {
  std::string word;
  int start = -1;
  int end = -1;

  WordPiece(std::string word, int start, int end)
      : word(std::move(word)), start(start), end(end) {}
};

struct DecodeResult {
  float score = -kFloatMax;
  std::string sentence;
  std::vector<WordPiece> word_pieces;

  static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
    return a.score > b.score;
  }
};

enum DecodeState {
  kEndBatch = 0x00,  // End of current decoding batch, normal case
  kEndpoint = 0x01,  // Endpoint is detected
  kEndFeats = 0x02,  // All feature is decoded
  kWaitFeats = 0x03  // Feat is not enough for one chunk inference, wait
};

// DecodeResource is thread safe, which can be shared for multiple
// decoding threads
struct DecodeResource {
  std::shared_ptr<AsrModel> model = nullptr;
  std::shared_ptr<fst::SymbolTable> symbol_table = nullptr;
  std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
  std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
  std::shared_ptr<ContextGraph> context_graph = nullptr;
  std::shared_ptr<PostProcessor> post_processor = nullptr;
};

// Torch ASR decoder
class AsrDecoder {
 public:
  AsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
             std::shared_ptr<DecodeResource> resource,
             const DecodeOptions& opts);
  // @param block: if true, block when feature is not enough for one chunk
  //               inference. Otherwise, return kWaitFeats.
  DecodeState Decode(bool block = true);
  void Rescoring();
  void Reset();
  void ResetContinuousDecoding();
  bool DecodedSomething() const {
    return !result_.empty() && !result_[0].sentence.empty();
  }

  // This method is used for time benchmark
  int num_frames_in_current_chunk() const {
    return num_frames_in_current_chunk_;
  }
  int frame_shift_in_ms() const {
    return model_->subsampling_rate() *
           feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  int feature_frame_shift_in_ms() const {
    return feature_pipeline_->config().frame_shift * 1000 /
           feature_pipeline_->config().sample_rate;
  }
  const std::vector<DecodeResult>& result() const { return result_; }

 private:
  DecodeState AdvanceDecoding(bool block = true);
  void AttentionRescoring();

  void UpdateResult(bool finish = false);

  std::shared_ptr<FeaturePipeline> feature_pipeline_;
  std::shared_ptr<AsrModel> model_;
  std::shared_ptr<PostProcessor> post_processor_;

  std::shared_ptr<fst::Fst<fst::StdArc>> fst_ = nullptr;
  // output symbol table
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  // e2e unit symbol table
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
  const DecodeOptions& opts_;
  // cache feature
  bool start_ = false;
  // For continuous decoding
  int num_frames_ = 0;
  int global_frame_offset_ = 0;
  const int time_stamp_gap_ = 100;  // timestamp gap between words in a sentence

  std::unique_ptr<SearchInterface> searcher_;
  std::unique_ptr<CtcEndpoint> ctc_endpointer_;

  int num_frames_in_current_chunk_ = 0;
  std::vector<DecodeResult> result_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(AsrDecoder);
};

}  // namespace wenet

#endif  // DECODER_ASR_DECODER_H_
