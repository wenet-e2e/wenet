
#ifndef FRONTEND_FBANK_CUDA_H_
#define FRONTEND_FBANK_CUDA_H_

#include "kaldifeat/csrc/feature-fbank.h"

namespace wenet {

class FbankCuda  {
 public:
  FbankCuda(int num_bins, int sample_rate) {
    fbank_opts_.mel_opts.num_bins = num_bins;
    fbank_opts_.frame_opts.samp_freq = sample_rate;
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.frame_opts.frame_shift_ms = 10.0;
    fbank_opts_.frame_opts.frame_length_ms = 25.0;
    fbank_opts_.device = torch::Device(torch::kCUDA, 0);
    fbank_ = std::make_shared<kaldifeat::Fbank>(fbank_opts_);
    device_ = torch::kCUDA;
  }

  torch::Tensor Compute(torch::Tensor wave_data) {
    return fbank_->ComputeFeatures(wave_data, 1.0f);
  }

  std::vector<torch::Tensor> Compute(
      const std::vector<std::vector<float>> &wave_data,
      std::vector<int> *num_frames) {
    const auto &frame_opts = fbank_->GetOptions().frame_opts;
    std::vector<int64_t> num_frames_vec;
    num_frames_vec.reserve(wave_data.size());

    std::vector<torch::Tensor> strided_vec;
    strided_vec.reserve(wave_data.size());

    for (const auto &w : wave_data) {
      torch::Tensor t = torch::from_blob(
          const_cast<float*>(w.data()),
          {static_cast<int>(w.size())}, torch::kFloat).to(device_);
      // t = t / 32768.0;
      torch::Tensor strided = kaldifeat::GetStrided(t, frame_opts);
      num_frames_vec.push_back(strided.size(0));
      num_frames->push_back(strided.size(0));
      strided_vec.emplace_back(std::move(strided));
    }

    torch::Tensor strided = torch::cat(strided_vec, 0);
    torch::Tensor features = fbank_->ComputeFeatures(strided, /*vtln_warp*/ 1.0f);
    auto ans = features.split_with_sizes(num_frames_vec, /*dim*/ 0);
    return ans;
  }

 private:
  kaldifeat::FbankOptions fbank_opts_;
  std::shared_ptr<kaldifeat::Fbank> fbank_;
  torch::DeviceType device_;

};

}  // namespace wenet

#endif  // FRONTEND_FBANK_CUDA_H_
