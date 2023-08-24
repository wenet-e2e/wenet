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

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xpu/runtime.h"
#include "xpu/xdnn.h"

#pragma once
namespace api = baidu::xpu::api;
template <typename T>
class XPUQunatData {
 public:
  XPUQunatData() : data_(nullptr), max_ptr_(nullptr) {}
  XPUQunatData(T* data, float* max_ptr) : data_(data), max_ptr_(max_ptr) {}
  T* data_{nullptr};
  float* max_ptr_{nullptr};
};

int vector_prod(std::vector<int> shape);
void add_separator_when_necessary(std::string& str);  // NOLINT

template <typename T, typename TW>
void conformer_test(const std::string& data_dir, const std::string& params_dir,
                    int threads_number, int dev_id);

template <typename T>
std::vector<T> Split(const std::string& str, const std::string& separator);

std::unordered_map<std::string, int> get_weights_lens(
    const std::string& file_path);
std::unordered_map<std::string, std::vector<int>> get_weights_shape(
    const std::string& file_path);

template <typename T>
std::vector<T> get_cpu_data(const std::string& file_path, int len);

template <typename T>
T* get_xpu_data(const std::string& file_path, int len);

template <typename T>
T* get_xpu_data(const std::string& data_name, const std::vector<T>& cpu_data);

template <typename TX, typename TY>
XPUQunatData<TY> get_xpu_quant_data(const std::string& file_path, int len);

template <typename TX, typename TY>
XPUQunatData<TY> get_xpu_quant_data(const std::string& data_name,
                                    const std::vector<TX>& cpu_data);

std::vector<int> get_all_ids(const std::string& dir_in);

void get_ln_param(const std::unordered_map<std::string, int>& weights_len_info,
                  const std::string& params_dir,
                  const std::string& ln_name_prefix,
                  const float*& ln_scale,  // NOLINT
                  const float*& ln_bias);  // NOLINT

template <typename T>
void print_vec(const std::vector<T>& data, const std::string& data_name);
template <typename T>
void print_cpu_data(const T* data, std::vector<int> shape, std::string name);
template <typename T>
void print_xpu_data(api::Context* ctx, const T* data, std::vector<int> shape,
                    std::string name);
template <typename T>
void print_xpu_data_all(api::Context* ctx, const T* data,
                        std::vector<int> shape, std::string name);

#define CHECK_RET(ret)                                    \
  if ((ret) != 0) {                                       \
    std::cout << "ERR" << __FILE__ << ":" << __LINE__     \
              << ", check failed, ret != 0" << std::endl; \
    std::exit(1);                                         \
  }
#define WRAPPER_CHECK_CTX(ctx) \
  if (ctx == nullptr) {        \
    return api::INVALID_PARAM; \
  }
#define WRAPPER_ASSERT_GT(ctx, expra, exprb) \
  if (!((expra) > (exprb))) {                \
    return api::INVALID_PARAM;               \
  }
#define WRAPPER_ASSERT_SUCCESS(ctx, ret) \
  if (!((ret) == api::SUCCESS)) {        \
    return ret;                          \
  }
