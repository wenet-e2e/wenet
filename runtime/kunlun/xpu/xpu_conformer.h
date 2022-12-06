// Copyright (c) 2022 KUNLUNXIN Inc.
//               2022 Han Qi (qihan@baidu.com)
//                    Hehe Pan (panhehe@baidu.com)
//                    Zikui Yan (yanzikui@baidu.com)
//                    Chaolin Li (lichaolin@baidu.com)
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

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xpu/runtime.h"
#include "xpu/xdnn.h"
#include "xpu_util.h"  // NOLINT
#pragma once

namespace api = baidu::xpu::api;
template <typename T, typename TW>
class ConformerEncoderParam {
 public:
  int layer_num;
  int fc_num_per_layer;
  int conv_num_per_layer;
  int ln_num_per_layer;
  int head_num;
  int head_dim;
  int ctc_dim;
  int ffn_factor;
  int beam_size;
  struct Embedding {
    int conv_num;
    int fc_num;
    int embed_dim;
  } emb_param;
  struct ConvBlock {
    bool is_casual;
    int kernel_size;
    int lorder;
    T padding;
  } conv_param;

  std::vector<const T*> pos_emb;
  std::vector<const TW*> emb_conv_w_list;
  std::vector<const float*> emb_conv_maxw_list;
  std::vector<const float*> emb_conv_bias_list;
  std::vector<const TW*> emb_fc_w_list;
  std::vector<const float*> emb_fc_maxw_list;
  std::vector<const float*> emb_fc_bias_list;

  std::vector<const TW*> conv_w_list;
  std::vector<const float*> conv_maxw_list;
  std::vector<const float*> conv_bias_list;

  std::vector<const float*> ln_scale_list;
  std::vector<const float*> ln_bias_list;

  std::vector<const TW*> fc_w_list;
  std::vector<const float*> fc_maxw_list;
  std::vector<const float*> fc_bias_list;

  std::vector<const TW*> attn_pos_w_list_;
  std::vector<const T*> attn_pos_w_list;
  std::vector<const float*> attn_pos_maxw_list;
  std::vector<const T*> attn_pos_uv_bias_list;

  const float* cmvn_istd{nullptr};
  const float* cmvn_mean{nullptr};
  const float* pe{nullptr};
  float* mask{nullptr};
};

template <typename T, typename TW>
class ConformerDecoderParam {
 public:
  int layer_num;
  int fc_num_per_layer;
  int ln_num_per_layer;

  int head_num;
  int head_dim;
  int vocab_size;
  int sos_id;
  int eos_id;
  int ignored_id;
  int beam_size;
  int max_token_num;
  int add_sos_num;
  int ffn_dim;

  const T* embed_table{nullptr};
  const T* pe{nullptr};
  std::vector<const TW*> fc_w_list;
  std::vector<const float*> fc_maxw_list;
  std::vector<const float*> fc_bias_list;
  std::vector<const float*> ln_scale_list;
  std::vector<const float*> ln_bias_list;
};

template <typename T>
static int64_t vec_prod(const std::vector<T>& data) {
  int len = data.size();
  if (len < 1) {
    return 0;
  }
  int64_t prod = data[0];
  for (int i = 1; i < len; ++i) {
    prod *= data[i];
  }
  return prod;
}

template <typename T>
static std::vector<const T*> get_w_list_from(
    const std::vector<XPUQunatData<T>>& quant_data_list) {
  int len = quant_data_list.size();
  std::vector<const T*> w_list(len, nullptr);
  for (int i = 0; i < len; ++i) {
    w_list[i] = quant_data_list[i].data_;
  }
  return w_list;
}

template <typename T>
static std::vector<const float*> get_w_maxptr_list_from(
    const std::vector<XPUQunatData<T>>& quant_data_list) {
  int len = quant_data_list.size();
  std::vector<const float*> w_maxptr_list(len, nullptr);
  for (int i = 0; i < len; ++i) {
    w_maxptr_list[i] = quant_data_list[i].max_ptr_;
  }
  return w_maxptr_list;
}

template <typename TW>
void get_fc_param(const std::unordered_map<std::string, int>& weights_len_info,
                  const std::string& params_dir,
                  const std::string& fc_name_prefix,
                  XPUQunatData<TW>& fc_w,                         // NOLINT
                  const float*& fc_bias, bool has_bias = true) {  // NOLINT
  const std::string fc_file_prefix = params_dir + fc_name_prefix;
  int wlen = weights_len_info.at(fc_name_prefix + "weight");
  fc_w = get_xpu_quant_data<float, TW>(fc_file_prefix + "weight", wlen);
  if (has_bias) {
    int blen = weights_len_info.at(fc_name_prefix + "bias");
    fc_bias = get_xpu_data<float>(fc_file_prefix + "bias", blen);
  } else {
    fc_bias = nullptr;
  }
}

template <typename TW>
void get_conv_param(
    const std::unordered_map<std::string, int>& weights_len_info,
    const std::string& params_dir, const std::string& conv_name_prefix,
    XPUQunatData<TW>& conv_w, const float*& conv_b,  // NOLINT
    bool has_bias = true) {                          // NOLINT
  std::string conv_file_prefix = params_dir + conv_name_prefix;
  int wlen = weights_len_info.at(conv_name_prefix + "weight");
  conv_w = get_xpu_quant_data<float, TW>(conv_file_prefix + "weight", wlen);
  if (has_bias) {
    int blen = weights_len_info.at(conv_name_prefix + "bias");
    conv_b = get_xpu_data<float>(conv_file_prefix + "bias", blen);
  } else {
    conv_b = nullptr;
  }
}

template <typename TW>
void get_fc_fused_param(
    const std::unordered_map<std::string, int>& weights_len_info,
    const std::string& params_dir,
    const std::vector<std::string> fc_name_prefixs,
    XPUQunatData<TW>& _fc_w,                      // NOLINT
    const float*& _fc_b, bool has_bias = true) {  // NOLINT
  // get cpu fc params
  std::vector<float> fc_ws;
  std::vector<float> fc_bs;
  for (int ids = 0; ids < fc_name_prefixs.size(); ids++) {
    std::string fc_file_prefix = params_dir + fc_name_prefixs[ids];
    int wlen = weights_len_info.at(fc_name_prefixs[ids] + "weight");
    std::vector<float> fc_w =
        get_cpu_data<float>(fc_file_prefix + "weight", wlen);
    std::vector<float> fc_b;
    if (has_bias) {
      int blen = weights_len_info.at(fc_name_prefixs[ids] + "bias");
      fc_b = get_cpu_data<float>(fc_file_prefix + "bias", blen);
    }
    fc_ws.insert(fc_ws.end(), fc_w.begin(), fc_w.end());
    fc_bs.insert(fc_bs.end(), fc_b.begin(), fc_b.end());
  }
  _fc_w = get_xpu_quant_data<float, TW>("fused_fc_weight", fc_ws);
  _fc_b = get_xpu_data<float>("fused_fc_bias", fc_bs);
}

template <typename TW>
void get_fc_ln_fused_param(
    const std::unordered_map<std::string, int>& weights_len_info,
    const std::string& params_dir,
    const std::vector<std::string> fc_name_prefixs,
    std::vector<std::string> ln_name_prefixs,
    XPUQunatData<TW>& _fc_w,                      // NOLINT
    const float*& _fc_b, bool has_bias = true) {  // NOLINT
  // get cpu fc params
  std::vector<float> fc_ws;
  std::vector<float> fc_bs;
  for (int ids = 0; ids < fc_name_prefixs.size(); ids++) {
    std::string fc_file_prefix = params_dir + fc_name_prefixs[ids];
    int wlen = weights_len_info.at(fc_name_prefixs[ids] + "weight");
    std::vector<float> fc_w =
        get_cpu_data<float>(fc_file_prefix + "weight", wlen);
    std::vector<float> fc_b;
    if (has_bias) {
      int blen = weights_len_info.at(fc_name_prefixs[ids] + "bias");
      fc_b = get_cpu_data<float>(fc_file_prefix + "bias", blen);
    }
    // get cpu ln params
    std::string ln_file_prefix = params_dir + ln_name_prefixs[ids];
    wlen = weights_len_info.at(ln_name_prefixs[ids] + "weight");
    int blen = weights_len_info.at(ln_name_prefixs[ids] + "bias");
    std::vector<float> ln_scale =
        get_cpu_data<float>(ln_file_prefix + "weight", wlen);
    std::vector<float> ln_bias =
        get_cpu_data<float>(ln_file_prefix + "bias", blen);
    int col = ln_scale.size();
    int row = static_cast<int>(fc_w.size()) / col;
    if (!has_bias) {
      fc_b.resize(row);
    }
    // get new fc_bias
    for (int i = 0; i < row; i++) {
      float b = has_bias ? fc_b[i] : 0.f;
      for (int j = 0; j < col; j++) {
        b += fc_w[i * col + j] * ln_bias[j];
      }
      fc_b[i] = b;
    }
    // get new fc_weight
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        fc_w[i * col + j] = fc_w[i * col + j] * ln_scale[j];
      }
    }
    fc_ws.insert(fc_ws.end(), fc_w.begin(), fc_w.end());
    fc_bs.insert(fc_bs.end(), fc_b.begin(), fc_b.end());
  }
  _fc_w = get_xpu_quant_data<float, TW>("fused_fc_weight", fc_ws);
  _fc_b = get_xpu_data<float>("fused_fc_bias", fc_bs);
}

template <typename TW>
void get_conv_bn_fused_param(
    const std::unordered_map<std::string, int>& weights_len_info,
    const std::string& params_dir, const std::string& conv_name_prefix,
    const std::string& bn_name_prefix, XPUQunatData<TW>& _conv_w,  // NOLINT
    const float*& _conv_b, bool has_bias = true) {                 // NOLINT
  // get cpu conv params
  std::string conv_file_prefix = params_dir + conv_name_prefix;
  int wlen = weights_len_info.at(conv_name_prefix + "weight");
  std::vector<float> conv_w =
      get_cpu_data<float>(conv_file_prefix + "weight", wlen);
  std::vector<float> conv_b;
  if (has_bias) {
    int blen = weights_len_info.at(conv_name_prefix + "bias");
    conv_b = get_cpu_data<float>(conv_file_prefix + "bias", blen);
  }
  // get cpu bn params
  std::string bn_file_prefix = params_dir + bn_name_prefix;
  wlen = weights_len_info.at(bn_name_prefix + "weight");
  int blen = weights_len_info.at(bn_name_prefix + "bias");
  int mlen = weights_len_info.at(bn_name_prefix + "running_mean");
  int vlen = weights_len_info.at(bn_name_prefix + "running_var");
  std::vector<float> bn_scale =
      get_cpu_data<float>(bn_file_prefix + "weight", wlen);
  std::vector<float> bn_bias =
      get_cpu_data<float>(bn_file_prefix + "bias", blen);
  std::vector<float> bn_mean =
      get_cpu_data<float>(bn_file_prefix + "running_mean", mlen);
  std::vector<float> bn_var =
      get_cpu_data<float>(bn_file_prefix + "running_var", vlen);
  // fuse conv, bn, new weight is conv_w, new bias is bn_bias
  int h = bn_scale.size();
  int w = static_cast<int>(conv_w.size()) / h;
  float eps = 1e-5f;  // assume eps is 1e-5;
  for (int i = 0; i < h; ++i) {
    bn_scale[i] = bn_scale[i] / std::sqrt(bn_var[i] + eps);
  }
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      conv_w[i * w + j] *= bn_scale[i];
    }
  }
  for (int i = 0; i < h; ++i) {
    float b = has_bias ? conv_b[i] : 0.f;
    bn_bias[i] += ((b - bn_mean[i]) * bn_scale[i]);
  }
  _conv_w = get_xpu_quant_data<float, TW>("fused_conv_weight", conv_w);
  _conv_b = get_xpu_data<float>("fused_conv_bias", bn_bias);
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<int>> read_cpu_data_from_file(
    const std::string& data_file_prefix, int shape_ndim) {
  std::vector<T> res_data;
  std::string data_file = data_file_prefix + ".dat";
  std::string shape_file = data_file_prefix + "_shape.txt";
  std::ifstream inF(shape_file);
  if (!inF) {
    std::cout << "ERR: open file failed! " << shape_file << std::endl;
    std::exit(1);
  }
  char useless;  // (16, 523, 80) or (160, 1)
  std::vector<int> inshape(shape_ndim, 0);
  if (shape_ndim == 3) {
    inF >> useless >> inshape[0] >> useless >> inshape[1] >> useless >>
        inshape[2] >> useless;
  } else if (shape_ndim == 2) {
    inF >> useless >> inshape[0] >> useless >> inshape[1] >> useless;
  } else if (shape_ndim == 1) {
    inF >> useless >> inshape[0] >> useless >> useless;
  } else {
    std::cout << "ERR: only support shape ndim == 1, 2 or 3, but got "
              << shape_ndim << std::endl;
    std::exit(1);
  }

  int data_len = vec_prod(inshape);
  res_data = get_cpu_data<T>(data_file, data_len);
  return std::make_tuple(res_data, inshape);
}

template <typename T>
static std::tuple<T*, std::vector<int>> read_xpu_data_from_file(
    const std::string& data_file_prefix, int shape_ndim) {
  auto cpu_data_info = read_cpu_data_from_file<T>(data_file_prefix, shape_ndim);
  T* xpu_data = get_xpu_data<T>(data_file_prefix, std::get<0>(cpu_data_info));
  return std::make_tuple(xpu_data, std::get<1>(cpu_data_info));
}

template <typename T>
static std::tuple<T*, std::vector<int>> create_mask_according_speech_length(
    const std::vector<int>& speech_length, int max_seqlen,
    void* xpu_stream = nullptr) {
  int batch = speech_length.size();
  int mask_len = batch * max_seqlen;
  int subsample_mask_len = batch * (((max_seqlen - 1) / 2 - 1) / 2);
  std::vector<T> mask_cpu(mask_len, 0);
  std::vector<T> subsample_mask_cpu(subsample_mask_len, 0);
  // create mask, equal to 'masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)'
  for (int b = 0; b < batch; ++b) {
    int curr_seqlen = speech_length[b];
    for (int idx = 0; idx < curr_seqlen; ++idx) {
      mask_cpu.at(b * max_seqlen + idx) = 1;
    }
  }
  // create subsample_mask, equal to 'x_mask[:, :, :-2:2][:, :, :-2:2]'
  int sub_seqlen = subsample_mask_len / batch;
  for (int b = 0; b < batch; ++b) {
    for (int idx = 0; idx < sub_seqlen; ++idx) {
      subsample_mask_cpu.at(b * sub_seqlen + idx) =
          mask_cpu.at(b * max_seqlen + idx * 4);
    }
  }
  // copy to xpu
  T* subsample_mask_xpu = nullptr;
  int r = xpu_malloc(reinterpret_cast<void**>(&subsample_mask_xpu),
                     subsample_mask_len * sizeof(T));
  if (r != 0) {
    std::cout << "ERR: xpu_malloc failed!" << std::endl;
    std::exit(1);
  }
  r = xpu_wait(xpu_stream);
  if (r != 0) {
    std::cout << "ERR: xpu_wait failed!" << std::endl;
    std::exit(1);
  }
  r = xpu_memcpy(subsample_mask_xpu, subsample_mask_cpu.data(),
                 subsample_mask_len * sizeof(T),
                 XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  if (r != 0) {
    std::cout << "ERR: xpu_memcpy failed!" << std::endl;
    std::exit(1);
  }

  std::vector<int> subsample_mask_shape{batch, 1, sub_seqlen};
  return std::make_tuple(subsample_mask_xpu, subsample_mask_shape);
}

template <typename T, typename TW>
int init_encoder_params(
    const std::string& params_dir,
    ConformerEncoderParam<T, TW>& encoder_param) {  // NOLINT
  std::unordered_map<std::string, int> weights_len_info =
      get_weights_lens(params_dir + "weights_info.txt");
  std::unordered_map<std::string, std::vector<int>> weights_shape_info =
      get_weights_shape(params_dir + "weights_info.txt");

  // model struct param
  auto& head_num = encoder_param.head_num;
  auto& head_dim = encoder_param.head_dim;
  auto& ffn_factor = encoder_param.ffn_factor;
  auto& conv_param = encoder_param.conv_param;
  auto& emb_param = encoder_param.emb_param;
  auto& ctc_dim = encoder_param.ctc_dim;
  auto& encoder_layer_num = encoder_param.layer_num;
  auto& fc_num_per_layer = encoder_param.fc_num_per_layer;
  auto& conv_num_per_layer = encoder_param.conv_num_per_layer;
  auto& ln_num_per_layer = encoder_param.ln_num_per_layer;
  encoder_layer_num = 12;
  fc_num_per_layer = 6;
  conv_num_per_layer = 3;
  ln_num_per_layer = 6;
  emb_param.conv_num = 2;
  emb_param.fc_num = 1;
  emb_param.embed_dim = 512;
  ffn_factor =
      weights_shape_info.at("encoder.encoders.0.feed_forward.w_1.weight")[0] /
      weights_shape_info.at("encoder.encoders.0.feed_forward.w_1.weight")[1];
  head_dim =
      weights_shape_info.at("encoder.encoders.0.self_attn.pos_bias_u")[1];
  head_num =
      weights_shape_info.at("encoder.encoders.0.self_attn.pos_bias_u")[0];
  conv_param.kernel_size = weights_shape_info.at(
      "encoder.encoders.0.conv_module.depthwise_conv.weight")[2];
  conv_param.lorder = conv_param.kernel_size - 1;
  conv_param.padding = 0.0;
  conv_param.is_casual = true;
  ctc_dim = weights_len_info.at("ctc.ctc_lo.bias");
  encoder_param.beam_size = 3;

  // init encoder cmvn
  auto& pe = encoder_param.pe;
  auto& cmvn_istd = encoder_param.cmvn_istd;
  auto& cmvn_mean = encoder_param.cmvn_mean;
  int pe_len = weights_len_info.at("encoder.pe");
  int mlen = weights_len_info.at("encoder.global_cmvn.mean");
  int ilen = weights_len_info.at("encoder.global_cmvn.istd");
  pe = get_xpu_data<float>(params_dir + "encoder.pe", pe_len);
  cmvn_mean =
      get_xpu_data<float>(params_dir + "encoder.global_cmvn.mean", mlen);
  cmvn_istd =
      get_xpu_data<float>(params_dir + "encoder.global_cmvn.istd", ilen);

  // init encoder embedding param
  std::vector<XPUQunatData<TW>> emb_conv_w_list;
  auto& emb_conv_bias_list = encoder_param.emb_conv_bias_list;
  std::vector<XPUQunatData<TW>> emb_fc_w_list;
  auto& emb_fc_bias_list = encoder_param.emb_fc_bias_list;
  emb_conv_w_list.resize(emb_param.conv_num);
  emb_conv_bias_list.resize(emb_param.conv_num);
  emb_fc_w_list.resize(emb_param.fc_num);
  emb_fc_bias_list.resize(emb_param.fc_num);
  for (int i = 0; i < emb_param.conv_num; ++i) {
    std::string conv_name_prefix =
        "encoder.embed.conv." + std::to_string(i * 2) + ".";
    get_conv_param<TW>(weights_len_info, params_dir, conv_name_prefix,
                       emb_conv_w_list[i], emb_conv_bias_list[i]);
  }
  get_fc_param<TW>(weights_len_info, params_dir, "encoder.embed.out.0.",
                   emb_fc_w_list[0], emb_fc_bias_list[0]);

  // encoder_param_layer
  int enc_fc_num = encoder_layer_num * fc_num_per_layer + 1;
  int enc_conv_num = encoder_layer_num * conv_num_per_layer;
  int enc_ln_num = encoder_layer_num * ln_num_per_layer + 1;

  std::vector<XPUQunatData<TW>> fc_w_list;
  auto& fc_bias_list = encoder_param.fc_bias_list;

  std::vector<XPUQunatData<TW>> conv_w_list;
  auto& conv_bias_list = encoder_param.conv_bias_list;

  auto& ln_scale_list = encoder_param.ln_scale_list;
  auto& ln_bias_list = encoder_param.ln_bias_list;

  std::vector<XPUQunatData<TW>> attn_pos_w_list;
  std::vector<const float*> attn_pos_uv_bias_list;
  // w_param need to be quanted & get maxw
  fc_w_list.resize(enc_fc_num);
  fc_bias_list.resize(enc_fc_num);
  conv_w_list.resize(enc_conv_num);
  conv_bias_list.resize(enc_conv_num);
  ln_scale_list.resize(enc_ln_num);
  ln_bias_list.resize(enc_ln_num);
  attn_pos_w_list.resize(encoder_layer_num);
  attn_pos_uv_bias_list.resize(encoder_layer_num *
                               2);  // pos_bias_u, pos_bias_v
  for (int i = 0; i < encoder_layer_num; ++i) {
    std::string enc_prefix = "encoder.encoders." + std::to_string(i) + ".";
    int fc_offset = i * fc_num_per_layer;
    int conv_offset = i * conv_num_per_layer;
    int ln_offset = i * ln_num_per_layer;
    // init FeedForwardParam macaron
    get_fc_param<TW>(weights_len_info, params_dir,
                     enc_prefix + "feed_forward_macaron.w_1.",
                     fc_w_list[fc_offset], fc_bias_list[fc_offset]);
    get_fc_param<TW>(weights_len_info, params_dir,
                     enc_prefix + "feed_forward_macaron.w_2.",
                     fc_w_list[fc_offset + 1], fc_bias_list[fc_offset + 1]);
    get_fc_fused_param<TW>(
        weights_len_info, params_dir,
        {enc_prefix + "self_attn.linear_q.", enc_prefix + "self_attn.linear_k.",
         enc_prefix + "self_attn.linear_v."},
        fc_w_list[fc_offset + 2], fc_bias_list[fc_offset + 2]);
    get_fc_param<TW>(
        weights_len_info, params_dir, enc_prefix + "self_attn.linear_out.",
        fc_w_list[fc_offset + 3], fc_bias_list[fc_offset + 3], true);
    // get pos w, pos u bias, pos v bias
    std::string pos_w_name = enc_prefix + "self_attn.linear_pos.weight";
    std::string pos_ubias_name = enc_prefix + "self_attn.pos_bias_u";
    std::string pos_vbias_name = enc_prefix + "self_attn.pos_bias_v";
    int pos_wlen = weights_len_info.at(pos_w_name);
    int pos_ublen = weights_len_info.at(pos_ubias_name);
    int pos_vblen = weights_len_info.at(pos_vbias_name);
    attn_pos_w_list[i] =
        get_xpu_quant_data<float, TW>(params_dir + pos_w_name, pos_wlen);
    attn_pos_uv_bias_list[i * 2] =
        get_xpu_data<float>(params_dir + pos_ubias_name, pos_ublen);
    attn_pos_uv_bias_list[i * 2 + 1] =
        get_xpu_data<float>(params_dir + pos_vbias_name, pos_vblen);
    // init ConvModuleParam
    get_conv_param<TW>(weights_len_info, params_dir,
                       enc_prefix + "conv_module.pointwise_conv1.",
                       conv_w_list[conv_offset], conv_bias_list[conv_offset],
                       true);
    get_conv_param<TW>(weights_len_info, params_dir,
                       enc_prefix + "conv_module.depthwise_conv.",
                       conv_w_list[conv_offset + 1],
                       conv_bias_list[conv_offset + 1], true);
    get_conv_param<TW>(weights_len_info, params_dir,
                       enc_prefix + "conv_module.pointwise_conv2.",
                       conv_w_list[conv_offset + 2],
                       conv_bias_list[conv_offset + 2], true);
    // init FeedForwardParam
    get_fc_param<TW>(weights_len_info, params_dir,
                     enc_prefix + "feed_forward.w_1.", fc_w_list[fc_offset + 4],
                     fc_bias_list[fc_offset + 4]);
    get_fc_param<TW>(weights_len_info, params_dir,
                     enc_prefix + "feed_forward.w_2.", fc_w_list[fc_offset + 5],
                     fc_bias_list[fc_offset + 5]);
    // init LayerNormParam
    get_ln_param(weights_len_info, params_dir, enc_prefix + "norm_ff_macaron.",
                 ln_scale_list[ln_offset], ln_bias_list[ln_offset]);
    get_ln_param(weights_len_info, params_dir, enc_prefix + "norm_mha.",
                 ln_scale_list[ln_offset + 1], ln_bias_list[ln_offset + 1]);
    get_ln_param(weights_len_info, params_dir, enc_prefix + "norm_conv.",
                 ln_scale_list[ln_offset + 2], ln_bias_list[ln_offset + 2]);
    get_ln_param(weights_len_info, params_dir, enc_prefix + "conv_module.norm.",
                 ln_scale_list[ln_offset + 3], ln_bias_list[ln_offset + 3]);
    get_ln_param(weights_len_info, params_dir, enc_prefix + "norm_ff.",
                 ln_scale_list[ln_offset + 4], ln_bias_list[ln_offset + 4]);
    get_ln_param(weights_len_info, params_dir, enc_prefix + "norm_final.",
                 ln_scale_list[ln_offset + 5], ln_bias_list[ln_offset + 5]);
  }
  get_ln_param(weights_len_info, params_dir, "encoder.after_norm.",
               ln_scale_list[enc_ln_num - 1], ln_bias_list[enc_ln_num - 1]);
  get_fc_param<TW>(weights_len_info, params_dir, "ctc.ctc_lo.",
                   fc_w_list[enc_fc_num - 1], fc_bias_list[enc_fc_num - 1]);
  /* get maxw && w */
  encoder_param.emb_conv_w_list = get_w_list_from<TW>(emb_conv_w_list);
  encoder_param.emb_conv_maxw_list =
      get_w_maxptr_list_from<TW>(emb_conv_w_list);
  encoder_param.emb_fc_w_list = get_w_list_from<TW>(emb_fc_w_list);
  encoder_param.emb_fc_maxw_list = get_w_maxptr_list_from<TW>(emb_fc_w_list);

  encoder_param.conv_w_list = get_w_list_from<TW>(conv_w_list);
  encoder_param.conv_maxw_list = get_w_maxptr_list_from<TW>(conv_w_list);

  encoder_param.fc_w_list = get_w_list_from<TW>(fc_w_list);
  encoder_param.fc_maxw_list = get_w_maxptr_list_from<TW>(fc_w_list);

  encoder_param.attn_pos_w_list_ = get_w_list_from<TW>(attn_pos_w_list);
  encoder_param.attn_pos_maxw_list =
      get_w_maxptr_list_from<TW>(attn_pos_w_list);
  /* prepare params */
  api::Context ctx_xpu(api::kXPU2);
  api::ctx_guard RAII_GUARD(&ctx_xpu);
  int ret = 0;
  int hidden_dim = head_num * head_dim;
  encoder_param.pos_emb.resize(encoder_layer_num);
  for (int i = 0; i < encoder_layer_num; i++) {
    ret = xpu_malloc((void**)&(encoder_param.pos_emb[i]),  // NOLINT
                     5000 * hidden_dim * sizeof(T));
    ret = api::fc_fusion<float, TW, T, int16_t>(
        &ctx_xpu, encoder_param.pe, encoder_param.attn_pos_w_list_[i],
        const_cast<T*>(encoder_param.pos_emb[i]), 5000, hidden_dim, hidden_dim,
        false, true, nullptr, encoder_param.attn_pos_maxw_list[i], nullptr,
        hidden_dim, hidden_dim, hidden_dim, 1.0f, 0.0f, nullptr,
        api::Activation_t::LINEAR);
  }
  for (int i = 0; i < encoder_layer_num; i++) {
    ret = api::scale<float>(
        &ctx_xpu, encoder_param.fc_bias_list[i * fc_num_per_layer + 1],
        const_cast<float*>(
            encoder_param.fc_bias_list[i * fc_num_per_layer + 1]),
        hidden_dim, true, 0.5f, 0.0f);
    ret = api::scale<float>(
        &ctx_xpu, encoder_param.fc_bias_list[i * fc_num_per_layer + 5],
        const_cast<float*>(
            encoder_param.fc_bias_list[i * fc_num_per_layer + 5]),
        hidden_dim, true, 0.5f, 0.0f);
  }
  for (int i = 0; i < attn_pos_uv_bias_list.size(); i++) {
    T* tmppos = nullptr;
    ret = xpu_malloc(reinterpret_cast<void**>(&tmppos), hidden_dim * sizeof(T));
    ret = api::cast_v2<float, T>(&ctx_xpu, attn_pos_uv_bias_list[i], tmppos,
                                 hidden_dim);
    encoder_param.attn_pos_uv_bias_list.push_back(tmppos);
  }
  return 0;
}

template <typename T, typename TW>
int init_decoder_params(
    const std::string& params_dir,
    ConformerDecoderParam<T, TW>& decoder_param) {  // NOLINT
  std::unordered_map<std::string, int> weights_len_info =
      get_weights_lens(params_dir + "weights_info.txt");

  // init DecoderLayerParam
  auto& decoder_layer_num = decoder_param.layer_num;
  auto& fc_num_per_layer = decoder_param.fc_num_per_layer;
  auto& ln_num_per_layer = decoder_param.ln_num_per_layer;
  std::vector<XPUQunatData<TW>> fc_w_list;
  auto& fc_bias_list = decoder_param.fc_bias_list;
  auto& ln_scale_list = decoder_param.ln_scale_list;
  auto& ln_bias_list = decoder_param.ln_bias_list;
  decoder_layer_num = 3;
  fc_num_per_layer = 8;
  ln_num_per_layer = 3;
  int dec_fc_num = decoder_layer_num * fc_num_per_layer + 1;
  int dec_ln_num = decoder_layer_num * ln_num_per_layer + 1;
  fc_w_list.resize(dec_fc_num);
  fc_bias_list.resize(dec_fc_num);
  ln_scale_list.resize(dec_ln_num);
  ln_bias_list.resize(dec_ln_num);
  decoder_param.head_num = 8;
  decoder_param.head_dim = 64;
  decoder_param.vocab_size = 5538;
  decoder_param.sos_id = 5537;
  decoder_param.eos_id = 5537;
  decoder_param.ignored_id = 2;
  decoder_param.beam_size = 3;
  decoder_param.max_token_num = 200;
  decoder_param.add_sos_num = 1;
  decoder_param.ffn_dim = 2048;
  auto att_dim = decoder_param.head_num * decoder_param.head_dim;

  // init EmbeddingParam
  std::string embed_table_name = "decoder.left_decoder.embed.0.weight";
  std::vector<float> embed_table_cpu = get_cpu_data<float>(
      params_dir + embed_table_name, weights_len_info.at(embed_table_name));
  std::vector<T> embed_table_cpu_t(embed_table_cpu.size(), 0);
  for (int i = 0; i < static_cast<int>(embed_table_cpu.size()); ++i) {
    embed_table_cpu_t[i] =
        static_cast<T>(embed_table_cpu[i] * std::sqrt(att_dim));
  }
  decoder_param.embed_table =
      get_xpu_data<T>(embed_table_name, embed_table_cpu_t);

  // init pe
  std::string pe_name = "encoder.pe";
  std::vector<float> pe_cpu =
      get_cpu_data<float>(params_dir + pe_name, weights_len_info.at(pe_name));
  std::vector<T> pe_cpu_t(pe_cpu.size(), 0);
  for (int i = 0; i < static_cast<int>(pe_cpu.size()); ++i) {
    pe_cpu_t[i] = static_cast<T>(pe_cpu[i]);
  }
  decoder_param.pe = get_xpu_data<T>(pe_name, pe_cpu_t);
  for (int i = 0; i < decoder_layer_num; ++i) {
    std::string dec_prefix =
        "decoder.left_decoder.decoders." + std::to_string(i) + ".";
    int offset = i * fc_num_per_layer;
    // init fc param
    // self attention qkv fc
    get_fc_fused_param<TW>(weights_len_info, params_dir,
                           {
                               dec_prefix + "self_attn.linear_q.",
                               dec_prefix + "self_attn.linear_k.",
                               dec_prefix + "self_attn.linear_v.",
                           },
                           fc_w_list[offset], fc_bias_list[offset], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "self_attn.linear_out.",
                     fc_w_list[offset + 1], fc_bias_list[offset + 1], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "src_attn.linear_q.", fc_w_list[offset + 2],
                     fc_bias_list[offset + 2], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "src_attn.linear_k.", fc_w_list[offset + 3],
                     fc_bias_list[offset + 3], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "src_attn.linear_v.", fc_w_list[offset + 4],
                     fc_bias_list[offset + 4], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "src_attn.linear_out.", fc_w_list[offset + 5],
                     fc_bias_list[offset + 5], true);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "feed_forward.w_1.", fc_w_list[offset + 6],
                     fc_bias_list[offset + 6]);
    get_fc_param<TW>(weights_len_info, params_dir,
                     dec_prefix + "feed_forward.w_2.", fc_w_list[offset + 7],
                     fc_bias_list[offset + 7]);
    // init ln param
    offset = i * ln_num_per_layer;
    get_ln_param(weights_len_info, params_dir, dec_prefix + "norm1.",
                 ln_scale_list[offset], ln_bias_list[offset]);
    get_ln_param(weights_len_info, params_dir, dec_prefix + "norm2.",
                 ln_scale_list[offset + 1], ln_bias_list[offset + 1]);
    get_ln_param(weights_len_info, params_dir, dec_prefix + "norm3.",
                 ln_scale_list[offset + 2], ln_bias_list[offset + 2]);
  }
  // init after ln
  get_ln_param(weights_len_info, params_dir, "decoder.left_decoder.after_norm.",
               ln_scale_list[dec_ln_num - 1], ln_bias_list[dec_ln_num - 1]);
  // init output layer fc
  get_fc_param<TW>(
      weights_len_info, params_dir, "decoder.left_decoder.output_layer.",
      fc_w_list[dec_fc_num - 1], fc_bias_list[dec_fc_num - 1], true);
  decoder_param.fc_w_list = get_w_list_from<TW>(fc_w_list);
  decoder_param.fc_maxw_list = get_w_maxptr_list_from<TW>(fc_w_list);
  return 0;
}

static int padding_target(std::vector<int>& hyps,      // NOLINT
                          std::vector<int>& hyps_len,  // NOLINT
                          int beam_size, int eos_id) {
  int max_target_len = *max_element(hyps_len.begin(), hyps_len.end());
  std::vector<int> pad(max_target_len * beam_size);
  int offset = 0;
  for (int i = 0; i < beam_size; i++) {
    for (int j = 0; j < max_target_len; j++) {
      pad[i * max_target_len + j] = j < hyps_len[i] ? hyps[j + offset] : eos_id;
    }
    offset += hyps_len[i];
  }
  hyps.swap(pad);
  return max_target_len;
}

namespace xpu {
namespace wenet {

template <typename T, typename TW, typename TGEMM>
int conformer_encoder_wenet(
    api::Context* ctx, float* x, const std::vector<int>& data_shape,
    T* encoder_out, T* ctc_probs,
    ConformerEncoderParam<T, TW>& param,  // NOLINT
    const std::tuple<float*, std::vector<int>>& xpu_mask_info);
template <typename T>
int ctc_prefix_beamsearch(api::Context* ctx, T* ctc_probs,
                          std::vector<int>& hyps,          // NOLINT
                          std::vector<int>& hyps_len,      // NOLINT
                          std::vector<float>& ctc_scores,  // NOLINT
                          int batch_size, int beam_size, int max_len,
                          int ctc_dim);

template <typename T, typename TW, typename TGEMM>
int conformer_decoder_wenet(api::Context* ctx, const T* x,
                            const std::vector<int32_t>& x_shape,
                            const float* x_mask, const int* padded_target,
                            const std::vector<int32_t>& target_shape,
                            float* character_scores,
                            const ConformerDecoderParam<T, TW>& param);
}  // namespace wenet
}  // namespace xpu
