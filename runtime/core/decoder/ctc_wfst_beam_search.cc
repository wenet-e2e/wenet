// Copyright (c) 2021 Mobvoi Inc (Binbin Zhang)
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


#include "decoder/ctc_wfst_beam_search.h"

#include <utility>

namespace wenet {

void DecodableTensorScaled::Reset() {
  num_frames_ready_ = 0;
  done_ = false;
  // Give an empty initialization, will throw error when
  // AcceptLoglikes is not called
  logp_.clear();
}

void DecodableTensorScaled::AcceptLoglikes(const std::vector<float>& logp) {
  ++num_frames_ready_;
  // TODO(Binbin Zhang): Avoid copy here
  logp_ = logp;
}

float DecodableTensorScaled::LogLikelihood(int32 frame, int32 index) {
  CHECK_GT(index, 0);
  CHECK_LT(frame, num_frames_ready_);
  return scale_ * logp_[index - 1];
}

bool DecodableTensorScaled::IsLastFrame(int32 frame) const {
  CHECK_LT(frame, num_frames_ready_);
  return done_ && (frame == num_frames_ready_ - 1);
}

int32 DecodableTensorScaled::NumIndices() const {
  LOG(FATAL) << "Not implement";
  return 0;
}

CtcWfstBeamSearch::CtcWfstBeamSearch(
    const fst::Fst<fst::StdArc>& fst, const CtcWfstBeamSearchOptions& opts,
    const std::shared_ptr<ContextGraph>& context_graph)
    : decodable_(opts.acoustic_scale),
      decoder_(fst, opts, context_graph),
      context_graph_(context_graph),
      opts_(opts) {
  Reset();
}

void CtcWfstBeamSearch::Reset() {
  num_frames_ = 0;
  decoded_frames_mapping_.clear();
  is_last_frame_blank_ = false;
  last_best_ = 0;
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  times_.clear();
  decodable_.Reset();
  decoder_.InitDecoding();
}

void CtcWfstBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
  if (0 == logp.size()) {
    return;
  }
  // Every time we get the log posterior, we decode it all before return
  for (int i = 0; i < logp.size(); i++) {
    float blank_score = std::exp(logp[i][0]);
    if (blank_score > opts_.blank_skip_thresh) {
      VLOG(3) << "skipping frame " << num_frames_ << " score " << blank_score;
      is_last_frame_blank_ = true;
      last_frame_prob_ = logp[i];
    } else {
      // Get the best symbol
      int cur_best =
          std::max_element(logp[i].begin(), logp[i].end()) - logp[i].begin();
      // Optional, adding one blank frame if we has skipped it in two same
      // symbols
      if (cur_best != 0 && is_last_frame_blank_ && cur_best == last_best_) {
        decodable_.AcceptLoglikes(last_frame_prob_);
        decoder_.AdvanceDecoding(&decodable_, 1);
        decoded_frames_mapping_.push_back(num_frames_ - 1);
        VLOG(2) << "Adding blank frame at symbol " << cur_best;
      }
      last_best_ = cur_best;

      decodable_.AcceptLoglikes(logp[i]);
      decoder_.AdvanceDecoding(&decodable_, 1);
      decoded_frames_mapping_.push_back(num_frames_);
      is_last_frame_blank_ = false;
    }
    num_frames_++;
  }
  // Get the best path
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  if (decoded_frames_mapping_.size() > 0) {
    inputs_.resize(1);
    outputs_.resize(1);
    likelihood_.resize(1);
    kaldi::Lattice lat;
    decoder_.GetBestPath(&lat, false);
    std::vector<int> alignment;
    kaldi::LatticeWeight weight;
    fst::GetLinearSymbolSequence(lat, &alignment, &outputs_[0], &weight);
    ConvertToInputs(alignment, &inputs_[0]);
    RemoveContinuousTags(&outputs_[0]);
    VLOG(3) << weight.Value1() << " " << weight.Value2();
    likelihood_[0] = -(weight.Value1() + weight.Value2());
  }
}

void CtcWfstBeamSearch::FinalizeSearch() {
  decodable_.SetFinish();
  decoder_.FinalizeDecoding();
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  times_.clear();
  if (decoded_frames_mapping_.size() > 0) {
    std::vector<kaldi::Lattice> nbest_lats;
    if (opts_.nbest == 1) {
      kaldi::Lattice lat;
      decoder_.GetBestPath(&lat, true);
      nbest_lats.push_back(std::move(lat));
    } else {
      // Get N-best path by lattice(CompactLattice)
      kaldi::CompactLattice clat;
      decoder_.GetLattice(&clat, true);
      kaldi::Lattice lat, nbest_lat;
      fst::ConvertLattice(clat, &lat);
      // TODO(Binbin Zhang): it's n-best word lists here, not character n-best
      fst::ShortestPath(lat, &nbest_lat, opts_.nbest);
      fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
    }
    int nbest = nbest_lats.size();
    inputs_.resize(nbest);
    outputs_.resize(nbest);
    likelihood_.resize(nbest);
    times_.resize(nbest);
    for (int i = 0; i < nbest; i++) {
      kaldi::LatticeWeight weight;
      std::vector<int> alignment;
      fst::GetLinearSymbolSequence(nbest_lats[i], &alignment, &outputs_[i],
                                   &weight);
      ConvertToInputs(alignment, &inputs_[i], &times_[i]);
      RemoveContinuousTags(&outputs_[i]);
      likelihood_[i] = -(weight.Value1() + weight.Value2());
    }
  }
}

void CtcWfstBeamSearch::ConvertToInputs(const std::vector<int>& alignment,
                                        std::vector<int>* input,
                                        std::vector<int>* time) {
  input->clear();
  if (time != nullptr) time->clear();
  for (int cur = 0; cur < alignment.size(); ++cur) {
    // ignore blank
    if (alignment[cur] - 1 == 0) continue;
    // merge continuous same label
    if (cur > 0 && alignment[cur] == alignment[cur - 1]) continue;

    input->push_back(alignment[cur] - 1);
    if (time != nullptr) {
      time->push_back(decoded_frames_mapping_[cur]);
    }
  }
}

void CtcWfstBeamSearch::RemoveContinuousTags(std::vector<int>* output) {
  if (context_graph_) {
    for (auto it = output->begin(); it != output->end();) {
      if (*it == context_graph_->start_tag_id() ||
          *it == context_graph_->end_tag_id()) {
        if (it + 1 != output->end() && *it == *(it + 1)) {
          it = output->erase(it);
          continue;
        }
      }
      ++it;
    }
  }
}

}  // namespace wenet
