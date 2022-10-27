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

#include "xpu_conformer.h"  // NOLINT
#include <chrono>
#include <mutex>
#include <thread>
#include <tuple>

namespace xpu {
namespace wenet {
const int X4_BEGIN = 8;
template <typename T, typename TW>
static int encoder_embed(api::Context* ctx_xpu, const float* x, T* y, int batch,
                         int max_seqlen, int seq_dim, int att_dim,
                         const ConformerEncoderParam<T, TW>& param) {
  api::ctx_guard RAII_GUARD(ctx_xpu);
  int ret = 0;
  int h_seqlen = (max_seqlen - 1) / 2;
  int q_seqlen = (h_seqlen - 1) / 2;
  int out_channels = att_dim;
  int h_dim = (seq_dim - 1) / 2;
  int q_dim = (h_dim - 1) / 2;

  float xscale = std::sqrt(att_dim);
  std::vector<int> sizes = {std::max(batch * max_seqlen * seq_dim,
                                     batch * out_channels * q_seqlen * q_dim),
                            batch * out_channels * h_seqlen * h_dim};
  std::vector<T*> ptrs;
  for (auto size_ind : sizes) {
    ptrs.push_back(RAII_GUARD.alloc<T>(size_ind));
  }

  auto& emb_conv_w_list = param.emb_conv_w_list;
  auto& emb_conv_maxw_list = param.emb_conv_maxw_list;
  auto& emb_conv_bias_list = param.emb_conv_bias_list;
  auto& emb_fc_w = param.emb_fc_w_list;
  auto& emb_fc_maxw = param.emb_fc_maxw_list;
  auto& emb_fc_bias = param.emb_fc_bias_list;

  ret =
      api::cast_v2<float, T>(ctx_xpu, x, ptrs[0], batch * max_seqlen * seq_dim);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = api::conv2d_fusion<T, TW, T, int16_t>(
      ctx_xpu, ptrs[0], emb_conv_w_list[0], ptrs[1], batch, 1, max_seqlen,
      seq_dim, out_channels, {3, 3}, {2, 2}, {0, 0}, {1, 1}, 1, nullptr,
      emb_conv_maxw_list[0], nullptr, true, emb_conv_bias_list[0], nullptr,
      api::Activation_t::RELU, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = api::conv2d_fusion<T, TW, T, int16_t>(
      ctx_xpu, ptrs[1], emb_conv_w_list[1], ptrs[0], batch, out_channels,
      h_seqlen, h_dim, out_channels, {3, 3}, {2, 2}, {0, 0}, {1, 1}, 1, nullptr,
      emb_conv_maxw_list[1], nullptr, true, emb_conv_bias_list[1], nullptr,
      api::Activation_t::RELU, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = api::transpose<T>(ctx_xpu, ptrs[0], ptrs[1],
                          {batch, out_channels, q_seqlen, q_dim}, {0, 2, 1, 3});
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = api::fc_fusion<T, TW, T, int16_t>(
      ctx_xpu, ptrs[1], emb_fc_w[0], ptrs[0], batch * q_seqlen, att_dim,
      out_channels * q_dim, false, true, nullptr, emb_fc_maxw[0], nullptr,
      out_channels * q_dim, out_channels * q_dim, att_dim, 1.0f, 0.0f,
      emb_fc_bias[0], api::Activation_t::LINEAR);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = api::scale<T>(ctx_xpu, ptrs[0], y, batch * q_seqlen * out_channels,
                      false, xscale, 0);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  ret = xpu_wait(ctx_xpu->xpu_stream);
  WRAPPER_ASSERT_SUCCESS(ctx_xpu, ret);
  return api::SUCCESS;
}

template <typename T, typename TW, typename TGEMM>
static int ffn(api::Context* ctx, int batch, int q_seqlen, int hidden_dim,
               bool with_endln, const T* x, T* y, int ln_begin, int fc_begin,
               std::vector<const float*> ln_scale_list,
               std::vector<const float*> ln_bias_list,
               std::vector<const TW*> fc_w_list,
               std::vector<const float*> fc_maxw_list,
               std::vector<const float*> fc_bias_list,
               std::vector<T*> mem_single, int ffn_factor) {
  api::ctx_guard RAII_GUARD(ctx);
  int ret = api::SUCCESS;
  std::unordered_map<std::string, T*> buf_mapping = {
      {"ffn_ln", mem_single[1]},          {"ffn_fc0", mem_single[X4_BEGIN]},
      {"tmp0", mem_single[X4_BEGIN + 1]}, {"tmp1", mem_single[X4_BEGIN]},
      {"ffn_fc1", mem_single[1]},
  };
  int ffn1_out_dim = hidden_dim * ffn_factor;
  int ffn2_input_dim = ffn1_out_dim;
  ret = api::layer_norm<T>(ctx, x, buf_mapping["ffn_ln"], batch * q_seqlen,
                           hidden_dim, 1e-5, ln_scale_list[ln_begin],
                           ln_bias_list[ln_begin], nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::fc_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["ffn_ln"], fc_w_list[fc_begin], buf_mapping["ffn_fc0"],
      batch * q_seqlen, ffn1_out_dim, hidden_dim, false, true, nullptr,
      fc_maxw_list[fc_begin], nullptr, hidden_dim, hidden_dim, ffn1_out_dim,
      1.0f, 0.0f, fc_bias_list[fc_begin], api::Activation_t::LINEAR);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::sigmoid<T>(ctx, buf_mapping["ffn_fc0"], buf_mapping["tmp0"],
                        batch * q_seqlen * hidden_dim * ffn_factor);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::mul<T>(ctx, buf_mapping["ffn_fc0"], buf_mapping["tmp0"],
                    buf_mapping["tmp1"],
                    batch * q_seqlen * hidden_dim * ffn_factor);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::fc_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["tmp1"], fc_w_list[fc_begin + 1], buf_mapping["ffn_fc1"],
      batch * q_seqlen, hidden_dim, ffn2_input_dim, false, true, nullptr,
      fc_maxw_list[fc_begin + 1], nullptr, ffn2_input_dim, ffn2_input_dim,
      hidden_dim, 0.5f, 0.0f, fc_bias_list[fc_begin + 1],
      api::Activation_t::LINEAR);
  if (with_endln) {
    ret = api::add_layer_norm_fusion<T>(
        ctx, x, buf_mapping["ffn_fc1"], y, batch * q_seqlen, hidden_dim, 1e-5,
        ln_scale_list[ln_begin + 1], ln_bias_list[ln_begin + 1]);
  } else {
    ret = api::add<T>(ctx, x, buf_mapping["ffn_fc1"], y,
                      batch * q_seqlen * hidden_dim);
  }
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return api::SUCCESS;
}

template <typename T, typename TW, typename TGEMM>
int wenet_encoder_layer(api::Context* ctx,
                        api::ctx_guard& RAII_GUARD,  // NOLINT
                        int batch, int q_seqlen, int hidden_dim, int ln_begin,
                        int fc_begin, int attn_pos_begin, int conv_begin,
                        const T* x, T* y,
                        ConformerEncoderParam<T, TW>& param,  // NOLINT
                        std::vector<T*>& mem_single,          // NOLINT
                        std::vector<T*>& mem_double,          // NOLINT
                        float* mem_float, float* mask_score) {
  WRAPPER_CHECK_CTX(ctx);
  int max_size = ctx->max_ptr_size();
  int ret = api::SUCCESS;
  std::unordered_map<std::string, T*> buf_mapping = {
      {"ffn0_out", mem_single[1]},
      {"swp0", mem_single[2]},
      {"swp1", mem_single[3]},
      {"matrix_bd_pre", mem_double[0]},
      {"soft_scores", mem_double[0]},
      {"qkv", mem_single[2]},
      {"qkv_add", mem_single[1]},
      {"conv_p1", mem_single[X4_BEGIN + 2]},
      {"conv_glu0", mem_single[X4_BEGIN + 3]},
      {"conv_glu1", mem_single[X4_BEGIN + 4]},
      {"conv_d1", mem_single[X4_BEGIN + 3]},
      {"conv_p2", mem_single[X4_BEGIN + 2]},
      {"conv_after", mem_single[0]},
  };

  auto ln_scale_list = param.ln_scale_list;
  auto ln_bias_list = param.ln_bias_list;

  auto fc_w_list = param.fc_w_list;
  auto fc_maxw_list = param.fc_maxw_list;
  auto fc_bias_list = param.fc_bias_list;

  auto attn_pos_w_list = param.attn_pos_w_list;
  auto attn_pos_maxw_list = param.attn_pos_maxw_list;
  auto attn_pos_uv_bias_list = param.attn_pos_uv_bias_list;

  auto conv_w_list = param.conv_w_list;
  auto conv_maxw_list = param.conv_maxw_list;
  auto conv_bias_list = param.conv_bias_list;

  auto kernel_size = param.conv_param.kernel_size;
  auto lorder = param.conv_param.lorder;
  auto padding = param.conv_param.padding;
  auto head_num = param.head_num;
  auto head_dim = param.head_dim;
  /*
  ** feed forward macaron-style module
  ** x = residual + 0.5*ff(x)
  */
  ret = ffn<T, TW, TGEMM>(ctx, batch, q_seqlen, hidden_dim, false, x,
                          buf_mapping["ffn0_out"], ln_begin, fc_begin,
                          ln_scale_list, ln_bias_list, fc_w_list, fc_maxw_list,
                          fc_bias_list, mem_single, param.ffn_factor);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  /*
  ** multi-headed self-attention module
  ** qkv_list[0-4]: q,k,v,qu,qv  mapping single[2-6]
  ** attn_pos_uv_bias_list : float -> float16
  ** q_pos_attention : get pos_emb before cal
  ** q_pos_attention : cal matrix_bd to qk_attention's mask ,when cal
  *qk_attention, mask will be added
  **/
  T* qkv_list[5] = {mem_single[6], mem_single[3], mem_single[4], mem_single[5],
                    mem_single[2]};
  ret = api::layer_norm<T>(ctx, buf_mapping["ffn0_out"], buf_mapping["swp0"],
                           batch * q_seqlen, hidden_dim, 1e-5,
                           ln_scale_list[ln_begin + 1],
                           ln_bias_list[ln_begin + 1], nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::fc_fusion_3c<T, TW, T, TGEMM>(
      ctx, buf_mapping["swp0"], fc_w_list[fc_begin + 2], qkv_list[0],
      qkv_list[1], qkv_list[2], batch * q_seqlen, hidden_dim * 3, hidden_dim,
      false, true, nullptr, fc_maxw_list[fc_begin + 2], nullptr, hidden_dim,
      hidden_dim, hidden_dim * 3, 1.0f, 0.0f, fc_bias_list[fc_begin + 2],
      api::Activation_t::LINEAR);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  for (int i = 0; i < 2; i++) {
    ret = api::broadcast_add<T>(
        ctx, qkv_list[0], attn_pos_uv_bias_list[attn_pos_begin * 2 + i],
        qkv_list[i + 3], {batch, q_seqlen, hidden_dim}, {1, 1, hidden_dim});
    WRAPPER_ASSERT_SUCCESS(ctx, ret);
  }
  int pos_emb_dim = 2 * q_seqlen - 1;
  T* pos_emb_sliced = RAII_GUARD.alloc<T>(pos_emb_dim * hidden_dim);
  ret = api::slice<T>(ctx, param.pos_emb[attn_pos_begin], pos_emb_sliced,
                      {5000, head_num, head_dim}, {0, 0, 0},
                      {pos_emb_dim, head_num, head_dim});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  int tmp_sliced_len = batch * head_num * q_seqlen * q_seqlen;
  float* tmp_mask = RAII_GUARD.alloc<float>(tmp_sliced_len);
  ret = api::q_pos_attention<T, T, T, TGEMM>(
      ctx, qkv_list[4], pos_emb_sliced, buf_mapping["matrix_bd_pre"], batch,
      q_seqlen, head_num, head_dim, 1.0f / std::sqrt(head_dim), nullptr,
      nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::slice<T>(ctx, buf_mapping["matrix_bd_pre"],
                      reinterpret_cast<T*>(mem_float),
                      {batch, head_num, q_seqlen, pos_emb_dim}, {0, 0, 0, 0},
                      {batch, head_num, q_seqlen, q_seqlen});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast_v2<T, float>(ctx, reinterpret_cast<T*>(mem_float), tmp_mask,
                               batch * head_num * q_seqlen * q_seqlen);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::broadcast_add<float>(ctx, tmp_mask, mask_score, mem_float,
                                  {batch, head_num, q_seqlen, q_seqlen},
                                  {batch, q_seqlen});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  api::QKVAttnParam loop_p(batch, q_seqlen, head_num, head_dim,
                           {batch, head_num, q_seqlen, q_seqlen},
                           api::Activation_t::LINEAR, -1, false, hidden_dim);
  float* qk_maxptr = RAII_GUARD.alloc<float>(max_size);
  ret = api::qk_attention<T, T, T, TGEMM>(
      ctx, qkv_list[3], qkv_list[1], buf_mapping["soft_scores"], nullptr,
      nullptr, qk_maxptr, loop_p, mem_float);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  float* qkv_maxptr = RAII_GUARD.alloc<float>(max_size);
  ret = api::qk_v_attention<T, T, T, TGEMM>(
      ctx, buf_mapping["soft_scores"], qkv_list[2], buf_mapping["qkv"],
      qk_maxptr, nullptr, qkv_maxptr, loop_p);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::fc_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["qkv"], fc_w_list[fc_begin + 3], buf_mapping["swp1"],
      batch * q_seqlen, hidden_dim, hidden_dim, false, true, qkv_maxptr,
      fc_maxw_list[fc_begin + 3], nullptr, hidden_dim, hidden_dim, hidden_dim,
      1.0f, 0.0f, fc_bias_list[fc_begin + 3], api::Activation_t::LINEAR);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::add<T>(ctx, buf_mapping["ffn0_out"], buf_mapping["swp1"],
                    buf_mapping["qkv_add"], batch * q_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  /*
  ** Conv conv_p1-conv_d1-conv_p2
  */
  ret = api::layer_norm<T>(ctx, buf_mapping["qkv_add"], buf_mapping["swp1"],
                           batch * q_seqlen, hidden_dim, 1e-5,
                           ln_scale_list[ln_begin + 2],
                           ln_bias_list[ln_begin + 2], nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::transpose<T>(ctx, buf_mapping["swp1"], buf_mapping["swp0"],
                          {batch, q_seqlen, hidden_dim}, {0, 2, 1});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  int pad_seqlen = q_seqlen;
  if (lorder > 0) {
    ret = api::pad<T>(ctx, buf_mapping["swp0"], buf_mapping["swp1"],
                      {batch, hidden_dim, q_seqlen}, {0, 0, lorder}, {0, 0, 0},
                      padding);
    WRAPPER_ASSERT_SUCCESS(ctx, ret);
    pad_seqlen += lorder;
  }
  ret = api::conv2d_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["swp1"], conv_w_list[conv_begin], buf_mapping["swp0"],
      batch, hidden_dim, 1, pad_seqlen, hidden_dim * 2, {1, 1}, {1, 1},
      {0, 0, 0, 0}, {1, 1}, 1, nullptr, conv_maxw_list[conv_begin], nullptr,
      true, conv_bias_list[conv_begin], nullptr, api::Activation_t::LINEAR,
      nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::split<T>(ctx, buf_mapping["swp0"],
                      {buf_mapping["conv_glu0"], buf_mapping["conv_glu1"]},
                      {batch, hidden_dim * 2, pad_seqlen},
                      {hidden_dim, hidden_dim}, 1);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::sigmoid(ctx, buf_mapping["conv_glu1"], buf_mapping["conv_glu1"],
                     batch * pad_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::mul(ctx, buf_mapping["conv_glu0"], buf_mapping["conv_glu1"],
                 buf_mapping["conv_p1"], batch * pad_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::conv1d_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["conv_p1"], conv_w_list[conv_begin + 1],
      buf_mapping["conv_d1"], batch, hidden_dim, pad_seqlen, hidden_dim,
      kernel_size, 1, {0}, 1, hidden_dim, nullptr,
      conv_maxw_list[conv_begin + 1], nullptr, true,
      conv_bias_list[conv_begin + 1], nullptr, api::Activation_t::LINEAR,
      nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);

  ret = api::transpose<T>(ctx, buf_mapping["conv_d1"], buf_mapping["swp0"],
                          {batch, hidden_dim, q_seqlen}, {0, 2, 1});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::layer_norm<T>(ctx, buf_mapping["swp0"], buf_mapping["swp1"],
                           batch * q_seqlen, hidden_dim, 1e-5,
                           ln_scale_list[ln_begin + 3],
                           ln_bias_list[ln_begin + 3], nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::sigmoid<T>(ctx, buf_mapping["swp1"], buf_mapping["swp0"],
                        batch * q_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::mul<T>(ctx, buf_mapping["swp0"], buf_mapping["swp1"],
                    buf_mapping["conv_p1"], batch * q_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::transpose<T>(ctx, buf_mapping["conv_p1"], buf_mapping["conv_d1"],
                          {batch, q_seqlen, hidden_dim}, {0, 2, 1});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::conv2d_fusion<T, TW, T, TGEMM>(
      ctx, buf_mapping["conv_d1"], conv_w_list[conv_begin + 2],
      buf_mapping["conv_p2"], batch, hidden_dim, 1, q_seqlen, hidden_dim,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, nullptr,
      conv_maxw_list[conv_begin + 2], nullptr, true,
      conv_bias_list[conv_begin + 2], nullptr, api::Activation_t::LINEAR,
      nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::transpose<T>(ctx, buf_mapping["conv_p2"], buf_mapping["swp0"],
                          {batch, hidden_dim, q_seqlen}, {0, 2, 1});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::add<T>(ctx, buf_mapping["swp0"], buf_mapping["qkv_add"],
                    buf_mapping["conv_after"], batch * q_seqlen * hidden_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  /*
  ** feed forward module
  ** x = residual + 0.5*ff(x)
  */
  ret = ffn<T, TW, TGEMM>(
      ctx, batch, q_seqlen, hidden_dim, true, buf_mapping["conv_after"], y,
      ln_begin + 4, fc_begin + 4, ln_scale_list, ln_bias_list, fc_w_list,
      fc_maxw_list, fc_bias_list, mem_single, param.ffn_factor);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return api::SUCCESS;
}

template <typename T, typename TW, typename TGEMM>
int conformer_encoder_wenet(
    api::Context* ctx, float* x, const std::vector<int>& data_shape,
    T* encoder_out, T* ctc_probs,
    ConformerEncoderParam<T, TW>& param,  // NOLINT
    const std::tuple<float*, std::vector<int>>& xpu_mask_info) {
  // Embedding -> Encoder_layer * N -> Layernorm -> Ctc_loss
  int ret = 0;
  int fc_num_per_layer = param.fc_num_per_layer;
  int conv_num_per_layer = param.conv_num_per_layer;
  int ln_num_per_layer = param.ln_num_per_layer;
  int ffn_factor = param.ffn_factor;
  int head_num = param.head_num;
  int head_dim = param.head_dim;
  int att_dim = head_num * head_dim;
  int ctc_dim = param.ctc_dim;
  int batch = data_shape[0];
  int max_seqlen = data_shape[1];
  int seq_dim = data_shape[2];
  int h_seqlen = (max_seqlen - 1) / 2;
  int q_seqlen = (h_seqlen - 1) / 2;

  WRAPPER_ASSERT_GT(ctx, param.layer_num, 0);
  WRAPPER_ASSERT_GT(ctx, batch, 0);
  WRAPPER_ASSERT_GT(ctx, head_num, 0);
  WRAPPER_ASSERT_GT(ctx, ctc_dim, 0);
  WRAPPER_ASSERT_GT(ctx, head_dim, 0);
  // Inital GM
  api::ctx_guard RAII_GUARD(ctx);
  std::vector<T*> mem_double;
  std::vector<T*> mem_single;
  int base_len = batch * (q_seqlen + 14) * (att_dim + 14);
  for (int i = 0; i < 8; i++) {
    mem_single.push_back(RAII_GUARD.alloc<T>(base_len));
  }
  mem_single.push_back(RAII_GUARD.alloc<T>(base_len * ffn_factor));
  mem_single.push_back(RAII_GUARD.alloc<T>(base_len * ffn_factor));
  mem_single.push_back(RAII_GUARD.alloc<T>(base_len * 4));
  mem_single.push_back(RAII_GUARD.alloc<T>(base_len * 4));
  mem_single.push_back(RAII_GUARD.alloc<T>(base_len * 2));
  mem_double.push_back(
      RAII_GUARD.alloc<T>(batch * head_num * q_seqlen * q_seqlen * 3));
  mem_double.push_back(
      RAII_GUARD.alloc<T>(batch * head_num * q_seqlen * q_seqlen));
  int ind_len = base_len * 6 + batch * param.head_num * q_seqlen * q_seqlen * 2;
  int lens =
      batch * param.head_num * q_seqlen * q_seqlen * sizeof(float) / sizeof(T);
  float* mem_float = RAII_GUARD.alloc<float>(lens);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  T* calx = mem_single[0];
  T* caly = mem_single[0];

  // embedding + mask
  float* emb = RAII_GUARD.alloc<float>(batch * max_seqlen * seq_dim);
  float* emb_nm = RAII_GUARD.alloc<float>(batch * max_seqlen * seq_dim);
  T* emb_fc = RAII_GUARD.alloc<T>(batch * q_seqlen * att_dim);
  ret = api::broadcast_sub<float>(ctx, x, param.cmvn_mean, emb, data_shape,
                                  {1, 1, 80});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::broadcast_mul<float>(ctx, emb, param.cmvn_istd, emb_nm, data_shape,
                                  {1, 1, 80});
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = encoder_embed<T, TW>(ctx, emb_nm, calx, batch, max_seqlen, seq_dim,
                             att_dim, param);
  float* mask_scores = RAII_GUARD.alloc<float>(batch * q_seqlen);
  ret = api::scale<float>(ctx, std::get<0>(xpu_mask_info), mask_scores,
                          batch * q_seqlen, false, 1e4, -1);
  CHECK_RET(ret);
  // encoder * N
  for (int i = 0; i < param.layer_num; i++) {
    int ln_begin = i * ln_num_per_layer;
    int fc_begin = i * fc_num_per_layer;
    int attn_pos_begin = i;
    int conv_begin = i * conv_num_per_layer;
    ret = wenet_encoder_layer<T, TW, int16_t>(
        ctx, RAII_GUARD, batch, q_seqlen, att_dim, ln_begin, fc_begin,
        attn_pos_begin, conv_begin, calx, caly, param, mem_single, mem_double,
        mem_float, mask_scores);
    WRAPPER_ASSERT_SUCCESS(ctx, ret);
    calx = caly;
  }
  // Final Layer_Norm
  int ln_begin = param.layer_num * param.ln_num_per_layer;
  int fc_begin = param.layer_num * param.fc_num_per_layer;
  auto final_ln_scale = param.ln_scale_list[ln_begin];
  auto final_ln_bias = param.ln_bias_list[ln_begin];
  ret = api::layer_norm(ctx, caly, encoder_out, batch * q_seqlen, att_dim, 1e-5,
                        final_ln_scale, final_ln_bias, nullptr, nullptr);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  // Ctc_Loss + log_sofmax
  auto ctc_fc_w = param.fc_w_list[fc_begin];
  auto ctc_fc_maxw = param.fc_maxw_list[fc_begin];
  auto ctc_fc_bias = param.fc_bias_list[fc_begin];
  float* ctc_buffer = RAII_GUARD.alloc<float>(batch * q_seqlen * ctc_dim);
  ret = api::fc_fusion<T, TW, float, TGEMM>(
      ctx, encoder_out, ctc_fc_w, ctc_buffer, batch * q_seqlen, ctc_dim,
      att_dim, false, true, nullptr, ctc_fc_maxw, nullptr, att_dim, att_dim,
      ctc_dim, 1.0f, 0.0f, ctc_fc_bias, api::Activation_t::LINEAR);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  float* softmax_out = RAII_GUARD.alloc<float>(batch * q_seqlen * ctc_dim);
  ret = api::softmax<float>(ctx, ctc_buffer, softmax_out,
                            {batch, q_seqlen, ctc_dim}, 2);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  float* log_out = RAII_GUARD.alloc<float>(batch * q_seqlen * ctc_dim);
  ret = api::log<float>(ctx, softmax_out, log_out, batch * q_seqlen * ctc_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast_v2<float, T>(ctx, log_out, ctc_probs,
                               batch * q_seqlen * ctc_dim);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  return api::SUCCESS;
}

#define INSTANTIATION_CONSFORMER_WENET(T, TW, TGEMM)          \
  template int conformer_encoder_wenet<T, TW, TGEMM>(         \
      api::Context*, float*, const std::vector<int>&, T*, T*, \
      ConformerEncoderParam<T, TW>&,                          \
      const std::tuple<float*, std::vector<int>>&);
INSTANTIATION_CONSFORMER_WENET(float16, int16_t, int16_t);

const float kFloatMax = std::numeric_limits<float>::max();
float logadd(std::vector<float> const& x) {
  float xmax = *max_element(x.begin(), x.end());
  if (xmax <= -kFloatMax) {
    return -kFloatMax;
  }
  float sum = 0.0;
  for (auto& it : x) {
    sum += std::exp(it - xmax);
  }
  return std::log(sum) + xmax;
}

struct PrefixScore {
  float s = -kFloatMax;
  float ns = -kFloatMax;
  float score() const { return logadd({s, ns}); }
  void check() const {
    std::cout << "score " << s << std::endl;
    std::cout << "nscore " << ns << std::endl;
  }
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
  return a.second.score() > b.second.score();
}

template <typename T>
int ctc_prefix_beamsearch(api::Context* ctx, T* ctc_probs,
                          std::vector<int>& hyps,                     // NOLINT
                          std::vector<int>& hyps_len,                 // NOLINT
                          std::vector<float>& ctc_scores, int batch,  // NOLINT
                          int beam_size, int max_len, int ctc_dim) {
  // 0. get topk
  api::ctx_guard RAII_GUARD(ctx);
  int data_len = batch * max_len * beam_size;
  int* topk_index_buf = RAII_GUARD.alloc<int>(data_len);
  float* topk_score_buf = RAII_GUARD.alloc<float>(data_len);
  float* logp = RAII_GUARD.alloc<float>(batch * max_len * ctc_dim);
  int ret =
      api::cast_v2<T, float>(ctx, ctc_probs, logp, batch * max_len * ctc_dim);
  ret = api::sorted_topk<float>(ctx, logp, topk_score_buf, topk_index_buf,
                                max_len, ctc_dim, beam_size, true);
  xpu_wait(ctx->xpu_stream);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  std::vector<int> topk_index(data_len);
  std::vector<float> topk_score(data_len);
  ret = xpu_memcpy(reinterpret_cast<void*>(&topk_index[0]), topk_index_buf,
                   data_len * sizeof(int), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  CHECK_RET(ret);
  ret = xpu_memcpy(reinterpret_cast<void*>(&topk_score[0]), topk_score_buf,
                   data_len * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  CHECK_RET(ret);
  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps;
  PrefixScore prefix_score;
  prefix_score.s = 0.0;
  prefix_score.ns = -kFloatMax;
  std::vector<int> empty;
  cur_hyps[empty] = prefix_score;
  for (int t = 0; t < max_len; ++t) {
    int offset = beam_size * t;
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
    // 1. Token passing
    for (int i = 0; i < beam_size; ++i) {
      int id = topk_index[i + offset];
      float prob = topk_score[i + offset];
      for (const auto& it : cur_hyps) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;
        if (id == 0) {
          // Case 0: *a + ε => *a
          PrefixScore& next_score = next_hyps[prefix];
          next_score.s = logadd(
              {next_score.s, prefix_score.s + prob, prefix_score.ns + prob});
          // Prefix not changed, copy the context from prefix.
          next_hyps[prefix] = next_score;
        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          PrefixScore& next_score = next_hyps[prefix];
          next_score.ns = logadd({next_score.ns, prefix_score.ns + prob});
          next_hyps[prefix] = next_score;
          // Case 2: *aε + a => *aa
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score1 = next_hyps[new_prefix];
          next_score1.ns = logadd({next_score1.ns, prefix_score.s + prob});
          next_hyps[new_prefix] = next_score1;
        } else {
          // Case 3: *a + b => *ab, *aε + b => *ab
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          next_score.ns = logadd(
              {next_score.ns, prefix_score.s + prob, prefix_score.ns + prob});
          next_hyps[new_prefix] = next_score;
        }
      }
    }
    // 2. Second beam prune, only keep top n best paths
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                              next_hyps.end());
    std::nth_element(arr.begin(), arr.begin() + beam_size, arr.end(),
                     PrefixScoreCompare);
    arr.resize(beam_size);
    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);
    // 3. Update cur_hyps and get new result
    cur_hyps.clear();
    for (int k = 0; k < beam_size; k++) {
      cur_hyps[arr[k].first] = arr[k].second;
    }
  }
  std::vector<std::pair<std::vector<int>, PrefixScore>> arr(cur_hyps.begin(),
                                                            cur_hyps.end());
  std::sort(arr.begin(), arr.end(), PrefixScoreCompare);
  int beam = 0;
  for (auto it : arr) {
    auto vec = it.first;
    hyps_len[beam] = vec.size();
    ctc_scores[beam] = it.second.score();
    hyps.insert(hyps.end(), vec.begin(), vec.end());
    beam++;
  }
  return api::SUCCESS;
}

template int ctc_prefix_beamsearch<float16>(
    api::Context* ctx, float16* logp,
    std::vector<int>& hyps,          // NOLINT
    std::vector<int>& hyps_len,      // NOLINT
    std::vector<float>& ctc_scores,  // NOLINT
    int batch, int beam_size, int max_len, int ctc_dim);

static int clip_cpu(int x, int min, int max) {
  if (x <= min) return min;
  if (x >= max) return max;
  return x;
}

static int add_sos_and_pad_ignored_id(
    api::Context* ctx, const int* target,
    std::vector<int>& pad_target,      // NOLINT
    std::vector<int>& pad_target_lod,  // NOLINT
    int batch_size, int target_seq_len, int max_target_seq_len, int eos_id,
    int ignored_id, int add_sos_num, int vocab_size) {
  int ret = -1;
  int target_data_len = batch_size * target_seq_len;
  std::vector<int> target_cpu(target_data_len);
  ret = xpu_wait(ctx->xpu_stream);
  ret = xpu_memcpy(reinterpret_cast<void*>(target_cpu.data()), target,
                   target_data_len * sizeof(int),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  for (int i = 0; i < batch_size; i++) {
    int valid_target_len = add_sos_num;
    for (int j = 0; j < target_seq_len; j++) {
      if (target_cpu[i * target_seq_len + j] == eos_id) {
        pad_target[i * max_target_seq_len + j + add_sos_num] = ignored_id;
      } else {
        pad_target[i * max_target_seq_len + j + add_sos_num] =
            clip_cpu(target_cpu[i * target_seq_len + j], 0, vocab_size);
        valid_target_len++;
      }
    }
    pad_target_lod[i + 1] = pad_target_lod[i] + valid_target_len;
  }
  return api::SUCCESS;
}

template <typename T, typename TW, typename TGEMM>
int conformer_decoder_wenet(api::Context* ctx, const T* x,
                            const std::vector<int32_t>& x_shape,
                            const float* x_mask, const int* padded_target,
                            const std::vector<int32_t>& target_shape,
                            float* character_scores,
                            const ConformerDecoderParam<T, TW>& param) {
  int layer_num = param.layer_num;
  int batch_size = x_shape[0];
  int beam_size = param.beam_size;
  int head_num = param.head_num;
  int head_dim = param.head_dim;
  int vocab_size = param.vocab_size;
  int dim = head_num * head_dim;
  int add_sos_num = param.add_sos_num;
  int new_bs = batch_size * beam_size;
  int sos_id = param.sos_id;
  int eos_id = param.eos_id;
  int ignored_id = param.ignored_id;
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_ASSERT_GT(ctx, layer_num, 0);
  WRAPPER_ASSERT_GT(ctx, batch_size, 0);
  WRAPPER_ASSERT_GT(ctx, head_num, 0);
  WRAPPER_ASSERT_GT(ctx, vocab_size, 0);
  WRAPPER_ASSERT_GT(ctx, dim, 0);

  api::ctx_guard RAII_GUARD(ctx);
  const int max_seq_len = x_shape[1];
  WRAPPER_ASSERT_GT(ctx, max_seq_len, 0);
  const int ffn1_out_dim = param.ffn_dim;
  // if ffn_act is glu
  const int ffn2_input_dim = ffn1_out_dim;
  const int d_k = dim / head_num;
  WRAPPER_ASSERT_GT(ctx, d_k, 0);
  int target_seq_len = target_shape[1];
  WRAPPER_ASSERT_GT(ctx, target_seq_len, 1);
  int max_target_seq_len = target_seq_len + add_sos_num;  // add sos
  WRAPPER_ASSERT_GT(ctx, max_seq_len, max_target_seq_len);

  int seqlen_sum = new_bs * max_seq_len;
  T* new_x = const_cast<T*>(x);
  int ret = -1;
  // get src_attn vsl input
  std::vector<float> cpu_mask_data(new_bs * max_seq_len, 0);
  std::vector<int> src_lod_vec(new_bs + 1, 0);
  ret = xpu_wait(ctx->xpu_stream);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = xpu_memcpy(reinterpret_cast<void*>(&cpu_mask_data.front()), x_mask,
                   new_bs * max_seq_len * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  for (int b = 1; b < src_lod_vec.size(); b++) {
    int curr_seqlen = 0;
    for (int idx = 0; idx < max_seq_len; idx++) {
      if (static_cast<int>(cpu_mask_data[idx]) == 1) {
        curr_seqlen++;
      }
    }
    src_lod_vec[b] = src_lod_vec[b - 1] + curr_seqlen;
  }
  api::VectorParam<int> src_qk_lods = {
      src_lod_vec.data(), static_cast<int>(src_lod_vec.size()), nullptr};
  src_qk_lods = src_qk_lods.to_xpu(RAII_GUARD);
  seqlen_sum = src_qk_lods.cpu[new_bs];

  T* broadcast_x = RAII_GUARD.alloc<T>(new_bs * max_seq_len * dim);
  ret = api::broadcast<T>(ctx, x, broadcast_x, {batch_size, max_seq_len, dim},
                          {new_bs, max_seq_len, dim});

  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  // add sos_id and pad ignored_id
  std::vector<int> real_target_cpu(max_target_seq_len * new_bs, sos_id);
  std::vector<int> real_target_lod(new_bs + 1, 0);

  ret = add_sos_and_pad_ignored_id(ctx, padded_target, real_target_cpu,
                                   real_target_lod, batch_size * beam_size,
                                   target_seq_len, max_target_seq_len, eos_id,
                                   ignored_id, add_sos_num, vocab_size);

  // get self/src QKVParam
  int target_seq_sum = real_target_lod[new_bs];
  api::VectorParam<int> self_qk_lods = {
      real_target_lod.data(), static_cast<int>(real_target_lod.size()),
      nullptr};
  self_qk_lods = self_qk_lods.to_xpu(RAII_GUARD);
  api::QKVAttnParam self_qkv_param(self_qk_lods, head_num, d_k,
                                   api::Activation_t::LINEAR);
  api::ConformerQKVParam src_qkv_param(self_qk_lods, src_qk_lods, head_num, d_k,
                                       false, -1);

  seqlen_sum = seqlen_sum > target_seq_sum ? seqlen_sum : target_seq_sum;
  std::vector<int> buf_sizes = {
      new_bs * max_target_seq_len *
          static_cast<int>(sizeof(int) / sizeof(T)),  // padded_target
      new_bs * max_target_seq_len * dim,              // embedding_out
      new_bs * max_target_seq_len * dim,              // mid_a
      new_bs * max_target_seq_len * dim,              // mid_b
      new_bs * max_target_seq_len *
          dim,  // attention_out, src_attention qk_v的结果
      new_bs * max_target_seq_len * dim,  // residual
      // ffn buffer
      new_bs * max_target_seq_len * ffn1_out_dim,    // ffn1_out
      new_bs * max_target_seq_len * ffn2_input_dim,  // ffn_glu_out
      new_bs * max_target_seq_len * ffn2_input_dim,  // ffn_glu_a
      new_bs * max_target_seq_len * ffn2_input_dim,  // ffn_glu_b
      new_bs * max_target_seq_len * ffn2_input_dim,  // ffn_glu_sigmoid
      // feature buffer
      new_bs * max_target_seq_len * dim * 3,  // feature_in buffer
      new_bs * max_target_seq_len * dim * 2,  // feature_out buffer
      new_bs * max_target_seq_len * 2,        // final_out
      seqlen_sum * dim,                       // q
      seqlen_sum * dim,                       // k
      seqlen_sum * dim,                       // v
      new_bs * max_seq_len * dim,             // src_x
      // attention buffer
      new_bs * max_seq_len * max_seq_len * dim,  // src_qk
  };
  std::vector<T*> buffer_ptrs(buf_sizes.size());
  for (int i = 0; i < buf_sizes.size(); i++) {
    buffer_ptrs[i] = RAII_GUARD.alloc<T>(buf_sizes[i]);
  }
  int b_id = 0;
  std::unordered_map<std::string, T*> buffer_map = {
      {"padded_target", buffer_ptrs[b_id++]},
      {"embedding_out", buffer_ptrs[b_id++]},
      {"mid_a", buffer_ptrs[b_id++]},
      {"mid_b", buffer_ptrs[b_id++]},
      {"attention_out", buffer_ptrs[b_id++]},
      {"residual", buffer_ptrs[b_id++]},
      {"ffn1_out", buffer_ptrs[b_id++]},
      {"ffn_glu_out", buffer_ptrs[b_id++]},
      {"ffn_glu_a", buffer_ptrs[b_id++]},
      {"ffn_glu_b", buffer_ptrs[b_id++]},
      {"ffn_glu_sigmoid", buffer_ptrs[b_id++]},
      {"feature_in", buffer_ptrs[b_id++]},
      {"feature_out", buffer_ptrs[b_id++]},
      {"final_out", buffer_ptrs[b_id++]},
      {"q", buffer_ptrs[b_id++]},
      {"k", buffer_ptrs[b_id++]},
      {"v", buffer_ptrs[b_id++]},
      {"src_x", buffer_ptrs[b_id++]},
      {"src_qk", buffer_ptrs[b_id++]},
  };
  // maxptr buffer
  int max_size = ctx->max_ptr_size();
  float* max_buffer = RAII_GUARD.alloc<float>(6 * max_size);
  float* max_x = max_buffer;
  float* max_q = max_buffer + max_size;
  float* max_k = max_buffer + 2 * max_size;
  float* max_v = max_buffer + 3 * max_size;
  float* max_qk = max_buffer + 4 * max_size;
  float* max_qkv = max_buffer + 5 * max_size;
  // copy pad_sos target to xpu
  int* new_paded_target = reinterpret_cast<int*>(buffer_map["padded_target"]);
  ret = api::do_host2device(ctx, real_target_cpu.data(), new_paded_target,
                            max_target_seq_len * new_bs * sizeof(int));
  T* embedding_out = buffer_map["embedding_out"];
  T* attention_out = buffer_map["attention_out"];
  T* mid_a = buffer_map["mid_a"];
  T* mid_b = buffer_map["mid_b"];
  T* q = buffer_map["q"];
  T* k = buffer_map["k"];
  T* v = buffer_map["v"];
  T* src_qk = buffer_map["src_qk"];
  T* residual = buffer_map["residual"];
  T* ffn1_out = buffer_map["ffn1_out"];
  T* ffn_glu_a = buffer_map["ffn_glu_a"];
  T* ffn_glu_b = buffer_map["ffn_glu_b"];
  T* ffn_glu_sigmoid = buffer_map["ffn_glu_sigmoid"];
  T* ffn_glu_out = buffer_map["ffn_glu_out"];
  // 1.1 embedding input: target{3,14} out:{3,14,512}
  ret =
      api::embedding<T, int>(ctx, param.embed_table, new_paded_target, residual,
                             vocab_size, dim, new_bs * max_target_seq_len, -1);
  float logit_scale = 1.0f;
  ret =
      api::scale<T>(ctx, residual, embedding_out,
                    new_bs * max_target_seq_len * dim, true, logit_scale, 0.0f);
  // 1.2 pos_embed, pos=[1, 5000, dim]
  ret = api::broadcast_add<T>(ctx, embedding_out, param.pe, residual,
                              {new_bs, max_target_seq_len, dim},
                              {1, max_target_seq_len, dim});
  // 2. decoder
  auto fc_weight_itr = param.fc_w_list.begin();
  auto fc_bias_itr = param.fc_bias_list.begin();
  auto fc_w_maxptr_itr = param.fc_maxw_list.begin();
  auto ln_scale_itr = param.ln_scale_list.begin();
  auto ln_bias_itr = param.ln_bias_list.begin();
  const float eps = 1e-5f;

  std::vector<float> mask_cpu(max_target_seq_len * max_target_seq_len, 0.0);
  const float kFloatMax = std::numeric_limits<float>::max();
  for (int j = 0; j < max_target_seq_len; j++) {
    for (int k = j + 1; k < max_target_seq_len; k++)
      mask_cpu[j * max_target_seq_len + k] = -kFloatMax;
  }
  float* mask_xpu;
  mask_xpu = reinterpret_cast<float*>(
      RAII_GUARD.alloc<float>(max_target_seq_len * max_target_seq_len));
  float* tg_mask;
  tg_mask = reinterpret_cast<float*>(RAII_GUARD.alloc<float>(
      new_bs * head_num * max_target_seq_len * max_target_seq_len));
  ret = xpu_memcpy(mask_xpu, reinterpret_cast<void*>(&mask_cpu[0]),
                   max_target_seq_len * max_target_seq_len * sizeof(float),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = api::broadcast<float>(
      ctx, mask_xpu, tg_mask, {1, 1, max_target_seq_len, max_target_seq_len},
      {new_bs, head_num, max_target_seq_len, max_target_seq_len});
  for (int j = 0; j < layer_num; j++) {
    // 2.1 self attention
    ret = api::layer_norm<T>(ctx, residual, mid_b, new_bs * max_target_seq_len,
                             dim, eps, *ln_scale_itr++, *ln_bias_itr++, nullptr,
                             nullptr);
    ret = api::fc_fusion_3c<T, TW, T, TGEMM>(
        ctx, mid_b, *fc_weight_itr++, q, k, v, target_seq_sum, dim * 3, dim,
        false, true, nullptr, *fc_w_maxptr_itr++, max_q, dim, dim, dim * 3,
        1.0f, 0.0f, *fc_bias_itr++, api::Activation_t::LINEAR);

    api::QKVAttnParam loop_p(
        new_bs, max_target_seq_len, head_num, d_k,
        {new_bs, head_num, max_target_seq_len, max_target_seq_len},
        api::Activation_t::LINEAR, -1, false, dim);

    ret = api::qk_attention<T, T, T, TGEMM>(ctx, q, k, src_qk, nullptr, nullptr,
                                            max_qk, loop_p, tg_mask);
    ret = api::qk_v_attention<T, T, T, TGEMM>(ctx, src_qk, v, mid_a, max_qk,
                                              nullptr, max_qkv, loop_p);
    // x + residual fused with fc
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, mid_a, *fc_weight_itr++, residual, new_bs * max_target_seq_len,
        dim, dim, false, true, nullptr, *fc_w_maxptr_itr++, nullptr, dim, dim,
        dim, 1.0f, 1.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
    // 2.2 src attention
    ret = api::layer_norm<T>(ctx, residual, mid_a, new_bs * max_target_seq_len,
                             dim, eps, *ln_scale_itr++, *ln_bias_itr++, nullptr,
                             nullptr);
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, mid_a, *fc_weight_itr++, mid_b, new_bs * max_target_seq_len, dim,
        dim, false, true, nullptr, *fc_w_maxptr_itr++, max_q, dim, dim, dim,
        1.0f, 0.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
    // get k,v use encoder_out
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, broadcast_x, *fc_weight_itr++, k, new_bs * max_seq_len, dim, dim,
        false, true, nullptr, *fc_w_maxptr_itr++, nullptr, dim, dim, dim, 1.0f,
        0.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, broadcast_x, *fc_weight_itr++, v, new_bs * max_seq_len, dim, dim,
        false, true, nullptr, *fc_w_maxptr_itr++, nullptr, dim, dim, dim, 1.0f,
        0.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
    ret = api::qk_attention<T, T, T, TGEMM>(ctx, mid_b, k, src_qk, nullptr,
                                            nullptr, max_qk, src_qkv_param);

    ret = api::qk_v_attention<T, T, T, TGEMM>(ctx, src_qk, v, mid_a, max_qk,
                                              nullptr, max_qkv, src_qkv_param);
    // x = x + residual fused with fc
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, mid_a, *fc_weight_itr++, residual, new_bs * max_target_seq_len,
        dim, dim, false, true, max_qkv, *fc_w_maxptr_itr++, nullptr, dim, dim,
        dim, 1.0f, 1.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
    // normalize before
    ret = api::layer_norm<T>(ctx, residual, mid_a, new_bs * max_target_seq_len,
                             dim, eps, *ln_scale_itr++, *ln_bias_itr++, nullptr,
                             nullptr);
    // ffn1
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, mid_a, *fc_weight_itr++, ffn1_out, new_bs * max_target_seq_len,
        ffn1_out_dim, dim, false, true, nullptr, *fc_w_maxptr_itr++, nullptr,
        dim, dim, ffn1_out_dim, 1.0, 0.0, *fc_bias_itr++,
        api::Activation_t::RELU);
    // ffn2
    ret = api::fc_fusion<T, TW, T, TGEMM>(
        ctx, ffn1_out, *fc_weight_itr++, residual, new_bs * max_target_seq_len,
        dim, ffn2_input_dim, false, true, nullptr, *fc_w_maxptr_itr++, nullptr,
        ffn2_input_dim, ffn2_input_dim, dim, 1.0, 1.0, *fc_bias_itr++,
        api::Activation_t::LINEAR);
  }

  ret =
      api::layer_norm(ctx, residual, mid_a, new_bs * max_target_seq_len, dim,
                      1e-5, *ln_scale_itr++, *ln_bias_itr++, nullptr, nullptr);
  int ctc_dim = param.vocab_size;
  ret = api::fc_fusion<T, TW, T, TGEMM>(
      ctx, mid_a, *fc_weight_itr++, mid_b, new_bs * max_target_seq_len, ctc_dim,
      dim, false, true, nullptr, *fc_w_maxptr_itr++, nullptr, dim, dim, ctc_dim,
      1.0f, 0.0f, *fc_bias_itr++, api::Activation_t::LINEAR);
  // log_softmax
  int data_len = new_bs * max_target_seq_len * ctc_dim;
  float* softmax_in = RAII_GUARD.alloc<float>(data_len);
  float* softmax_out = RAII_GUARD.alloc<float>(data_len);
  float* log_out = RAII_GUARD.alloc<float>(data_len);
  ret = api::cast_v2<T, float>(ctx, mid_b, softmax_in, data_len);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::softmax<float>(ctx, softmax_in, softmax_out,
                            {new_bs, max_target_seq_len, ctc_dim}, 2);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::log<float>(ctx, softmax_out, character_scores, data_len);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);

  return api::SUCCESS;
}

template int conformer_decoder_wenet<float16, int16_t, int16_t>(
    api::Context* ctx, const float16* x, const std::vector<int32_t>& x_shape,
    const float* x_mask, const int* padded_target,
    const std::vector<int32_t>& target_shape, float* character_scores,
    const ConformerDecoderParam<float16, int16_t>& param);

}  // namespace wenet
}  // namespace xpu
