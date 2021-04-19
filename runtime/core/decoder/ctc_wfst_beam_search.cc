// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include "decoder/ctc_wfst_beam_search.h"

namespace wenet {

void DecodableTensorScaled::Reset() {
  offset_ = 0;
  num_frames_ready_ = 0;
  done_ = false;
  // Give an empty initialization, will throw error when
  // AcceptPosterior is not called
  logp_ = torch::zeros({1});
}

void DecodableTensorScaled::AcceptPosterior(const torch::Tensor& logp) {
  CHECK_EQ(logp.dim(), 2);
  offset_ = num_frames_ready_;
  num_frames_ready_ += logp.size(0);
  // TODO(Binbin Zhang): Avoid copy here
  logp_ = logp;
  accessor_.reset(new torch::TensorAccessor<float, 2>(
      logp_.data_ptr<float>(), logp_.sizes().data(), logp_.strides().data()));
}

float DecodableTensorScaled::LogLikelihood(int32 frame, int32 index) {
  CHECK(accessor_ != nullptr);
  CHECK_GT(index, 0);
  CHECK_LE(index, logp_.size(1));
  CHECK_GE(frame, offset_);
  CHECK_LT(frame, num_frames_ready_);
  return scale_ * (*accessor_)[frame - offset_][index - 1];
}

bool DecodableTensorScaled::IsLastFrame(int32 frame) const {
  CHECK_LT(frame, num_frames_ready_);
  return done_ && (frame == num_frames_ready_ - 1);
}

int32 DecodableTensorScaled::NumIndices() const {
  CHECK_GT(logp_.size(0), 0);
  return logp_.size(1);
}

CtcWfstBeamSearch::CtcWfstBeamSearch(const fst::Fst<fst::StdArc>& fst,
                                     const CtcWfstBeamSearchOptions& opts)
    : decodable_(opts.acoustic_scale), decoder_(fst, opts), opts_(opts) {
  Reset();
}

void CtcWfstBeamSearch::Reset() {
  inputs_.clear();
  outputs_.clear();
  likelihood_.clear();
  times_.clear();
  decodable_.Reset();
  decoder_.InitDecoding();
}

void CtcWfstBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  if (0 == logp.size(0)) {
    return;
  }
  // Every time we get the log posterior, we decode it all before return
  decodable_.AcceptPosterior(logp);
  decoder_.AdvanceDecoding(&decodable_, logp.size(0));
  // Get the best path
  kaldi::Lattice lat;
  decoder_.GetBestPath(&lat, false);
  inputs_.resize(1);
  outputs_.resize(1);
  likelihood_.resize(1);
  inputs_[0].clear();
  outputs_[0].clear();
  std::vector<int> alignment;
  kaldi::LatticeWeight weight;
  fst::GetLinearSymbolSequence(lat, &alignment, &outputs_[0], &weight);
  ConvertToInputs(alignment, &inputs_[0]);
  likelihood_[0] = weight.Value1();
  LOG(INFO) << weight.Value1() << " " << weight.Value2();
}

void CtcWfstBeamSearch::FinalizeSearch() {
  decodable_.SetFinish();
  decoder_.FinalizeDecoding();
  // Get N-best path by lattice(CompactLattice)
  kaldi::CompactLattice clat;
  decoder_.GetLattice(&clat, true);
  kaldi::Lattice lat, nbest_lat;
  fst::ConvertLattice(clat, &lat);
  // TODO(Binbin Zhang): it's n-best word lists here, not character n-best
  fst::ShortestPath(lat, &nbest_lat, opts_.nbest);
  std::vector<kaldi::Lattice> nbest_lats;
  fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
  int nbest = nbest_lats.size();
  inputs_.resize(nbest);
  outputs_.resize(nbest);
  likelihood_.resize(nbest);
  for (int i = 0; i < nbest; i++) {
    inputs_[i].clear();
    outputs_[i].clear();
    kaldi::LatticeWeight weight;
    std::vector<int> alignment;
    fst::GetLinearSymbolSequence(nbest_lats[i], &alignment, &outputs_[i],
                                 &weight);
    ConvertToInputs(alignment, &inputs_[i]);
    likelihood_[i] = weight.Value1();
  }
}

void CtcWfstBeamSearch::ConvertToInputs(const std::vector<int>& alignment,
                                        std::vector<int>* input) {
  input->clear();
  for (size_t i = 0; i < alignment.size(); i++) {
    if (alignment[i] - 1 > 0) {
      input->push_back(alignment[i] - 1);
    }
  }
}

}  // namespace wenet
