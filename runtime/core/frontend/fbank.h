// Copyright (c) 2017 Personal (Binbin Zhang)
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

#ifndef FRONTEND_FBANK_H_
#define FRONTEND_FBANK_H_

#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "frontend/fft.h"
#include "utils/log.h"

#define S16_TO_FLOAT_SCALE 32768

namespace wenet {

// This code is based on kaldi Fbank implementation, please see
// https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.cc

enum class WindowType {
  Povey,
  Hanning,
};

enum class MelType {
  HTK,
  Slaney,
};

enum class NormalizationType {
  KALDI,
  Whisper,
};

enum class LogBase {
  BaseE,
  Base10,
};

class Fbank {
 public:
  Fbank(int num_bins, int sample_rate, int frame_length, int frame_shift,
        float low_freq = 20, bool pre_emphasis = true,
        bool scaled_float_as_input = false,
        float log_floor = std::numeric_limits<float>::epsilon(),
        LogBase log_base = LogBase::BaseE,
        WindowType window_type = WindowType::Povey,
        MelType mel_type = MelType::HTK,
        NormalizationType norm_type = NormalizationType::KALDI)
      : num_bins_(num_bins),
        sample_rate_(sample_rate),
        frame_length_(frame_length),
        frame_shift_(frame_shift),
        use_log_(true),
        remove_dc_offset_(true),
        generator_(0),
        distribution_(0, 1.0),
        dither_(0.0),
        pre_emphasis_(pre_emphasis),
        scaled_float_as_input_(scaled_float_as_input),
        log_floor_(log_floor),
        log_base_(log_base),
        norm_type_(norm_type) {
    fft_points_ = UpperPowerOfTwo(frame_length_);
    // generate bit reversal table and trigonometric function table
    const int fft_points_4 = fft_points_ / 4;
    bitrev_.resize(fft_points_);
    sintbl_.resize(fft_points_ + fft_points_4);
    make_sintbl(fft_points_, sintbl_.data());
    make_bitrev(fft_points_, bitrev_.data());

    int num_fft_bins = fft_points_ / 2;
    float fft_bin_width = static_cast<float>(sample_rate_) / fft_points_;
    int high_freq = sample_rate_ / 2;
    float mel_low_freq = MelScale(low_freq, mel_type);
    float mel_high_freq = MelScale(high_freq, mel_type);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);
    bins_.resize(num_bins_);
    center_freqs_.resize(num_bins_);

    for (int bin = 0; bin < num_bins; ++bin) {
      float left_mel = mel_low_freq + bin * mel_freq_delta,
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
      center_freqs_[bin] = InverseMelScale(center_mel, mel_type);
      std::vector<float> this_bin(num_fft_bins);
      int first_index = -1, last_index = -1;
      for (int i = 0; i < num_fft_bins; ++i) {
        float freq = (fft_bin_width * i);  // Center frequency of this fft
        // bin.
        float mel = MelScale(freq, mel_type);
        if (mel > left_mel && mel < right_mel) {
          float weight;
          if (mel_type == MelType::HTK) {
            if (mel <= center_mel)
              weight = (mel - left_mel) / (center_mel - left_mel);
            else if (mel > center_mel)
              weight = (right_mel - mel) / (right_mel - center_mel);
          } else if (mel_type == MelType::Slaney) {
            if (mel <= center_mel) {
              weight = (InverseMelScale(mel, mel_type) -
                        InverseMelScale(left_mel, mel_type)) /
                       (InverseMelScale(center_mel, mel_type) -
                        InverseMelScale(left_mel, mel_type));
              weight *= 2.0 / (InverseMelScale(right_mel, mel_type) -
                               InverseMelScale(left_mel, mel_type));
            } else if (mel > center_mel) {
              weight = (InverseMelScale(right_mel, mel_type) -
                        InverseMelScale(mel, mel_type)) /
                       (InverseMelScale(right_mel, mel_type) -
                        InverseMelScale(center_mel, mel_type));
              weight *= 2.0 / (InverseMelScale(right_mel, mel_type) -
                               InverseMelScale(left_mel, mel_type));
            }
          }
          this_bin[i] = weight;
          if (first_index == -1) first_index = i;
          last_index = i;
        }
      }
      CHECK(first_index != -1 && last_index >= first_index);
      bins_[bin].first = first_index;
      int size = last_index + 1 - first_index;
      bins_[bin].second.resize(size);
      for (int i = 0; i < size; ++i) {
        bins_[bin].second[i] = this_bin[first_index + i];
      }
    }
    InitWindow(window_type);
  }

  void set_use_log(bool use_log) { use_log_ = use_log; }

  void set_remove_dc_offset(bool remove_dc_offset) {
    remove_dc_offset_ = remove_dc_offset;
  }

  void set_dither(float dither) { dither_ = dither; }

  int num_bins() const { return num_bins_; }

  void InitWindow(WindowType window_type) {
    window_.resize(frame_length_);
    if (window_type == WindowType::Povey) {
      // povey window
      double a = M_2PI / (frame_length_ - 1);
      for (int i = 0; i < frame_length_; ++i)
        window_[i] = pow(0.5 - 0.5 * cos(a * i), 0.85);
    } else if (window_type == WindowType::Hanning) {
      // periodic hanning window
      double a = M_2PI / (frame_length_);
      for (int i = 0; i < frame_length_; ++i)
        window_[i] = 0.5 * (1.0 - cos(i * a));
    }
  }

  static inline float InverseMelScale(float mel_freq,
                                      MelType mel_type = MelType::HTK) {
    if (mel_type == MelType::HTK) {
      return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
    } else if (mel_type == MelType::Slaney) {
      float f_min = 0.0;
      float f_sp = 200.0 / 3.0;
      float min_log_hz = 1000.0;
      float freq = f_min + f_sp * mel_freq;
      float min_log_mel = (min_log_hz - f_min) / f_sp;
      float logstep = logf(6.4) / 27.0;
      if (mel_freq >= min_log_mel) {
        return min_log_hz * expf(logstep * (mel_freq - min_log_mel));
      } else {
        return freq;
      }
    }
  }

  static inline float MelScale(float freq, MelType mel_type = MelType::HTK) {
    if (mel_type == MelType::HTK) {
      return 1127.0f * logf(1.0f + freq / 700.0f);
    } else if (mel_type == MelType::Slaney) {
      float f_min = 0.0;
      float f_sp = 200.0 / 3.0;
      float min_log_hz = 1000.0;
      float mel = (freq - f_min) / f_sp;
      float min_log_mel = (min_log_hz - f_min) / f_sp;
      float logstep = logf(6.4) / 27.0;
      if (freq >= min_log_hz) {
        return min_log_mel + logf(freq / min_log_hz) / logstep;
      } else {
        return mel;
      }
    }
  }

  static int UpperPowerOfTwo(int n) {
    return static_cast<int>(pow(2, ceil(log(n) / log(2))));
  }

  // pre emphasis
  void PreEmphasis(float coeff, std::vector<float>* data) const {
    if (coeff == 0.0) return;
    for (int i = data->size() - 1; i > 0; i--)
      (*data)[i] -= coeff * (*data)[i - 1];
    (*data)[0] -= coeff * (*data)[0];
  }

  // Apply window on data in place
  void ApplyWindow(std::vector<float>* data) const {
    CHECK_GE(data->size(), window_.size());
    for (size_t i = 0; i < window_.size(); ++i) {
      (*data)[i] *= window_[i];
    }
  }

  void WhisperNorm(std::vector<std::vector<float>>* feat,
                   float max_mel_engery) {
    int num_frames = feat->size();
    for (int i = 0; i < num_frames; ++i) {
      for (int j = 0; j < num_bins_; ++j) {
        float energy = (*feat)[i][j];
        if (energy < max_mel_engery - 8) energy = max_mel_engery - 8;
        energy = (energy + 4.0) / 4.0;
        (*feat)[i][j] = energy;
      }
    }
  }

  // Compute fbank feat, return num frames
  int Compute(const std::vector<float>& wave,
              std::vector<std::vector<float>>* feat) {
    int num_samples = wave.size();

    if (num_samples < frame_length_) return 0;
    int num_frames = 1 + ((num_samples - frame_length_) / frame_shift_);
    feat->resize(num_frames);
    std::vector<float> fft_real(fft_points_, 0), fft_img(fft_points_, 0);
    std::vector<float> power(fft_points_ / 2);

    float max_mel_engery = std::numeric_limits<float>::min();

    for (int i = 0; i < num_frames; ++i) {
      std::vector<float> data(wave.data() + i * frame_shift_,
                              wave.data() + i * frame_shift_ + frame_length_);

      if (scaled_float_as_input_) {
        for (int j = 0; j < frame_length_; ++j) {
          data[j] = data[j] / S16_TO_FLOAT_SCALE;
        }
      }

      // optional add noise
      if (dither_ != 0.0) {
        for (size_t j = 0; j < data.size(); ++j)
          data[j] += dither_ * distribution_(generator_);
      }
      // optinal remove dc offset
      if (remove_dc_offset_) {
        float mean = 0.0;
        for (size_t j = 0; j < data.size(); ++j) mean += data[j];
        mean /= data.size();
        for (size_t j = 0; j < data.size(); ++j) data[j] -= mean;
      }

      if (pre_emphasis_) {
        PreEmphasis(0.97, &data);
      }
      ApplyWindow(&data);
      // copy data to fft_real
      memset(fft_img.data(), 0, sizeof(float) * fft_points_);
      memset(fft_real.data() + frame_length_, 0,
             sizeof(float) * (fft_points_ - frame_length_));
      memcpy(fft_real.data(), data.data(), sizeof(float) * frame_length_);
      fft(bitrev_.data(), sintbl_.data(), fft_real.data(), fft_img.data(),
          fft_points_);
      // power
      for (int j = 0; j < fft_points_ / 2; ++j) {
        power[j] = fft_real[j] * fft_real[j] + fft_img[j] * fft_img[j];
        power[j] = power[j];
      }

      (*feat)[i].resize(num_bins_);
      // cepstral coefficients, triangle filter array

      for (int j = 0; j < num_bins_; ++j) {
        float mel_energy = 0.0;
        int s = bins_[j].first;
        for (size_t k = 0; k < bins_[j].second.size(); ++k) {
          mel_energy += bins_[j].second[k] * power[s + k];
        }
        // optional use log
        if (use_log_) {
          if (mel_energy < log_floor_) mel_energy = log_floor_;

          if (log_base_ == LogBase::BaseE)
            mel_energy = logf(mel_energy);
          else if (log_base_ == LogBase::Base10)
            mel_energy = log10(mel_energy);
        }
        if (max_mel_engery < mel_energy) max_mel_engery = mel_energy;
        (*feat)[i][j] = mel_energy;
      }
    }
    if (norm_type_ == NormalizationType::Whisper)
      WhisperNorm(feat, max_mel_engery);

    // ReadVecFromCSV(feat, std::string("feat_std.csv"));
    return num_frames;
  }

 private:
  int num_bins_;
  int sample_rate_;
  int frame_length_, frame_shift_;
  int fft_points_;
  bool use_log_;
  bool remove_dc_offset_;
  bool pre_emphasis_;
  bool scaled_float_as_input_;
  float log_floor_;
  LogBase log_base_;
  NormalizationType norm_type_;

  std::vector<float> center_freqs_;
  std::vector<std::pair<int, std::vector<float>>> bins_;
  std::vector<float> window_;
  std::default_random_engine generator_;
  std::normal_distribution<float> distribution_;
  float dither_;

  // bit reversal table
  std::vector<int> bitrev_;
  // trigonometric function table
  std::vector<float> sintbl_;
};

}  // namespace wenet

#endif  // FRONTEND_FBANK_H_
