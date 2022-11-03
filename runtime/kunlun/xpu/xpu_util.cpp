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

#include "xpu_util.h"  // NOLINT

template <typename T>
static double vec_sum(const std::vector<T>& data) {
  double res = 0;
  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    res += static_cast<double>(data[i]);
  }
  return res;
}

int vector_prod(std::vector<int> shape) {
  int accumlate = 1;
  for (auto a : shape) {
    accumlate *= a;
  }
  return accumlate;
}
void add_separator_when_necessary(std::string& str) {  // NOLINT
  int len = str.size();
  char ch = '/';
  if (str[len - 1] != ch) {
    str.append(1, ch);
  }
}

template <typename T>
static std::string print_vec(const std::vector<T>& data) {
  std::stringstream ss;
  const int dump_len = data.size() > 8 ? 8 : data.size();
  std::vector<T> dump_data(dump_len, 0);
  int half_dump_len = dump_len / 2;
  std::copy(data.cbegin(), data.cbegin() + half_dump_len, dump_data.begin());
  std::copy(data.cend() - (dump_len - half_dump_len), data.cend(),
            dump_data.begin() + half_dump_len);
  for (int i = 0; i < dump_len - 1; ++i) {
    ss << dump_data[i] << ", ";
    if ((i + 1) == dump_len / 2) {
      ss << " ... ";
    }
  }
  ss << dump_data[dump_len - 1];
  return ss.str();
}

template <typename T>
static T parse_string(const std::string& str) {
  return str;
}

template <>
float parse_string(const std::string& str) {
  return std::stof(str);
}
template <>
double parse_string(const std::string& str) {
  return std::stod(str);
}
template <>
int parse_string(const std::string& str) {
  return std::stoi(str);
}
template <>
int64_t parse_string(const std::string& str) {
  return std::stoll(str);
}

template <typename T>
std::vector<T> Split(const std::string& str, const std::string& separator) {
  std::vector<T> res;
  std::string::size_type pos1, pos2;
  pos1 = str.find_first_not_of(separator);
  pos2 = str.find(separator, pos1);
  while (std::string::npos != pos1 && std::string::npos != pos2) {
    res.emplace_back(parse_string<T>(str.substr(pos1, pos2 - pos1)));
    pos1 = str.find_first_not_of(separator, pos2);
    pos2 = str.find(separator, pos1);
  }
  if (std::string::npos != pos1 && pos1 < str.length()) {
    res.emplace_back(parse_string<T>(str.substr(pos1)));
  }
  return res;
}

std::unordered_map<std::string, int> get_weights_lens(
    const std::string& file_path) {
  std::unordered_map<std::string, int> res;
  std::ifstream inF(file_path, std::ifstream::in);
  if (inF) {
    // std::cout << "read success from " << file_path << std::endl;
    std::string buffer;
    while (std::getline(inF, buffer)) {
      std::vector<std::string> weight_info = Split<std::string>(buffer, ":");
      std::string w_name = weight_info[0];
      int w_len = std::stoi(weight_info[3]);
      res.insert(std::make_pair(w_name, w_len));
    }
  } else {
    std::cout << "ERR: read failed, " << file_path << std::endl;
    std::exit(1);
  }

  return res;
}

std::unordered_map<std::string, std::vector<int>> get_weights_shape(
    const std::string& file_path) {
  std::unordered_map<std::string, std::vector<int>> res;
  std::ifstream inF(file_path, std::ifstream::in);
  if (inF) {
    // std::cout << "read success from " << file_path << std::endl;
    std::string buffer;
    while (std::getline(inF, buffer)) {
      std::vector<std::string> weight_info = Split<std::string>(buffer, ":");
      std::string w_name = weight_info[0];
      std::string w_shape_str = weight_info[2];  // example: (512, 1, 3, 3)
      std::string w_shape_str_without_bracket(
          w_shape_str.begin() + 1,
          w_shape_str.end() - 1);  // example: 512, 1, 3, 3
      std::vector<int> w_shape = Split<int>(w_shape_str_without_bracket, ",");
      res.insert(std::make_pair(w_name, w_shape));
    }
  } else {
    std::cout << "ERR: read failed, " << file_path << std::endl;
    std::exit(1);
  }

  return res;
}

template <typename T>
std::vector<T> get_cpu_data(const std::string& file_path, int len) {
  std::vector<T> result(len, 0);
  std::ifstream inF(file_path, std::ifstream::binary);
  if (!inF) {
    std::cout << "ERR: std::ifstream init failed! " << file_path << std::endl;
    std::exit(1);
  }
  if (inF.read(reinterpret_cast<char*>(result.data()), len * sizeof(T))) {
    // std::cout << "read success from " << file_path << std::endl;
  } else {
    std::cout << "ERR: something wrong: " << file_path << ", len=" << len
              << std::endl;
    std::exit(1);
  }
  return result;
}

template std::vector<float> get_cpu_data<float>(const std::string&, int len);
template std::vector<float16> get_cpu_data<float16>(const std::string&,
                                                    int len);
template std::vector<int64_t> get_cpu_data<int64_t>(const std::string&,
                                                    int len);
template std::vector<int> get_cpu_data<int>(const std::string&, int len);

template <typename T>
T* get_xpu_data(const std::string& data_name, const std::vector<T>& cpu_data) {
  int len = cpu_data.size();
#ifdef TEST_DEBUG
  std::cout << "DEBUG: file_path=" << data_name << ", len=" << len
            << ", vec_sum=" << vec_sum(cpu_data)
            << ", details: " << print_vec(cpu_data) << std::endl;
#endif

  T* xpu_data = nullptr;
  int r = xpu_malloc(reinterpret_cast<void**>(&xpu_data), len * sizeof(T));
  if (r != 0) {
    std::cout << "ERR: xpu_malloc failed! " << data_name << std::endl;
    std::exit(1);
  }

  r = xpu_wait();
  if (r != 0) {
    std::cout << "ERR: xpu_wait failed!" << std::endl;
    std::exit(1);
  }
  r = xpu_memcpy(xpu_data, cpu_data.data(), len * sizeof(T),
                 XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  if (r != 0) {
    std::cout << "ERR: xpu_memcpy failed! " << data_name << std::endl;
    std::exit(1);
  }

#ifdef TEST_DEBUG
  std::cout << "DEBUG: xpu_data=" << xpu_data << std::endl;
#endif

  return xpu_data;
}

template float* get_xpu_data(const std::string&, const std::vector<float>&);
template float16* get_xpu_data(const std::string&, const std::vector<float16>&);
template int64_t* get_xpu_data(const std::string&, const std::vector<int64_t>&);
template int* get_xpu_data(const std::string&, const std::vector<int>&);

template <typename T>
T* get_xpu_data(const std::string& file_path, int len) {
  std::vector<T> cpu_data = get_cpu_data<T>(file_path, len);
  return get_xpu_data<T>(file_path, cpu_data);
}

template float* get_xpu_data<float>(const std::string&, int);
template float16* get_xpu_data<float16>(const std::string&, int);
template int64_t* get_xpu_data<int64_t>(const std::string&, int);
template int* get_xpu_data<int>(const std::string&, int);

template <typename TX, typename TY>
std::vector<TY> quant_cpu(const std::vector<TX>& cpu_data) {
  int len = cpu_data.size();
  std::vector<TY> cpu_quant_data(len, 0);
  api::Context ctx(api::kCPU);
  int r = api::quantization<TX, TY>(&ctx, cpu_data.data(),
                                    cpu_quant_data.data(), len, nullptr);
  if (r != 0) {
    std::cout << "ERR: quantization failed!" << std::endl;
    std::exit(1);
  }
  return cpu_quant_data;
}

template <>
std::vector<float> quant_cpu<float, float>(const std::vector<float>& cpu_data) {
  return cpu_data;
}

template <typename TX, typename TY>
XPUQunatData<TY> get_xpu_quant_data(const std::string& data_name,
                                    const std::vector<TX>& cpu_data) {
  XPUQunatData<TY> xpu_quant_data;

  int len = cpu_data.size();
  // quant
  std::vector<TY> cpu_quant_data = quant_cpu<TX, TY>(cpu_data);
  // findmax
  float abs_max = 1e-30f;
  if (std::is_same<TX, float>::value || std::is_same<TX, float16>::value) {
    for (int i = 0; i < len; ++i) {
      float abs_val = std::fabs(static_cast<float>(cpu_data[i]));
      abs_max = std::max<float>(abs_max, abs_val);
    }
  }

  constexpr int max_ptr_len = 6;  // for xpu2
  std::vector<float> cpu_max(max_ptr_len, abs_max);
  // xpu malloc
  TY* xpu_data = nullptr;
  float* xpu_max_ptr = nullptr;
  int r = xpu_malloc(reinterpret_cast<void**>(&xpu_data), len * sizeof(TY));
  if (r != 0) {
    std::cout << "ERR: xpu_malloc failed! " << data_name << std::endl;
    std::exit(1);
  }
  r = xpu_malloc(reinterpret_cast<void**>(&xpu_max_ptr),
                 max_ptr_len * sizeof(float));
  if (r != 0) {
    std::cout << "ERR: xpu_malloc failed! " << data_name << std::endl;
    std::exit(1);
  }

#ifdef TEST_DEBUG
  std::cout << "DEBUG: file_path=" << data_name << ", len=" << len
            << ", data vec_sum=" << vec_sum(cpu_data)
            << ", quant_data vec_sum=" << vec_sum(cpu_quant_data)
            << ", details: " << print_vec(cpu_quant_data) << std::endl;
#endif
  r = xpu_wait();
  if (r != 0) {
    std::cout << "ERR: xpu_wait failed!" << std::endl;
    std::exit(1);
  }
  // xpu memcpy
  r = xpu_memcpy(xpu_data, cpu_quant_data.data(), len * sizeof(TY),
                 XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  if (r != 0) {
    std::cout << "ERR: xpu_memcpy failed!" << std::endl;
    std::exit(1);
  }
#ifdef TEST_DEBUG
  std::cout << "DEBUG: max is " << print_vec(cpu_max) << std::endl;
#endif
  r = xpu_memcpy(xpu_max_ptr, cpu_max.data(), max_ptr_len * sizeof(float),
                 XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  if (r != 0) {
    std::cout << "ERR: xpu_malloc failed!" << std::endl;
    std::exit(1);
  }

#ifdef TEST_DEBUG
  std::cout << "DEBUG: xpu_data=" << xpu_data << ", xpu_max_ptr=" << xpu_max_ptr
            << std::endl;
#endif
  xpu_quant_data.data_ = xpu_data;
  xpu_quant_data.max_ptr_ = xpu_max_ptr;
  return xpu_quant_data;
}

template XPUQunatData<float> get_xpu_quant_data<float, float>(
    const std::string&, const std::vector<float>&);
template XPUQunatData<int16_t> get_xpu_quant_data<float, int16_t>(
    const std::string&, const std::vector<float>&);

template <typename TX, typename TY>
XPUQunatData<TY> get_xpu_quant_data(const std::string& file_path, int len) {
  std::vector<TX> cpu_data = get_cpu_data<TX>(file_path, len);
  return get_xpu_quant_data<TX, TY>(file_path, cpu_data);
}

template XPUQunatData<float> get_xpu_quant_data<float, float>(
    const std::string&, int);
template XPUQunatData<int16_t> get_xpu_quant_data<float, int16_t>(
    const std::string&, int);

std::vector<int> get_all_ids(const std::string& dir_in) {
  std::vector<int> ids;
  std::set<int> ids_set;
  struct stat s;
  stat(dir_in.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    return ids;
  }
  DIR* open_dir = opendir(dir_in.c_str());
  if (nullptr == open_dir) {
    return ids;
  }
  dirent* p = nullptr;
  while ((p = readdir(open_dir)) != nullptr) {
    if (p->d_name[0] != '.') {
      std::string filename = std::string(p->d_name);
      int end_pos = filename.find('_');

      int qid = std::stoi(filename.substr(0, end_pos));
      ids_set.insert(qid);
    }
  }
  closedir(open_dir);
  ids.resize(ids_set.size());
  ids.assign(ids_set.begin(), ids_set.end());
  return ids;
}

void get_ln_param(const std::unordered_map<std::string, int>& weights_len_info,
                  const std::string& params_dir,
                  const std::string& ln_name_prefix,
                  const float*& ln_scale,   // NOLINT
                  const float*& ln_bias) {  // NOLINT
  std::string ln_file_prefix = params_dir + ln_name_prefix;
  int wlen = weights_len_info.at(ln_name_prefix + "weight");
  int blen = weights_len_info.at(ln_name_prefix + "bias");
  ln_scale = get_xpu_data<float>(ln_file_prefix + "weight", wlen);
  ln_bias = get_xpu_data<float>(ln_file_prefix + "bias", blen);
}

template <typename T>
void print_xpu_data_all(api::Context* ctx, const T* data,
                        std::vector<int> shape, std::string name) {
  int data_len = vector_prod(shape);
  std::vector<T> cpu_data(data_len);
  xpu_wait(ctx->xpu_stream);
  xpu_memcpy(reinterpret_cast<void**>(&cpu_data.front()), data,
             data_len * sizeof(T), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  std::cout << name;
  std::cout << " shape:";
  for (auto i : shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  int row = 1;
  int col = shape.back();
  if (shape.size() >= 2) {
    row = data_len / col;
  }
  T* cpu_data_ptr = &cpu_data.front();
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << *(cpu_data_ptr + i * col + j) << " ";
    }
    std::cout << std::endl;
  }
}
template <typename T>
void print_xpu_data(api::Context* ctx, const T* data, std::vector<int> shape,
                    std::string name) {
  int data_len = vector_prod(shape);

  std::vector<T> cpu_data(data_len);
  xpu_memcpy(reinterpret_cast<void*>(&cpu_data.front()), data,
             data_len * sizeof(T), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  std::cout << name;
  std::cout << " shape:";
  for (auto i : shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  if (data_len > 1000) {
    double mean = 0;
    for (auto val : cpu_data) {
      mean += static_cast<double>(val);
    }
    mean /= data_len;
    std::cout << "mean=" << mean << std::endl;
    std::cout << "details: ";
    for (int i = 0; i < 8; ++i) {
      std::cout << cpu_data[i] << " ";
    }
    std::cout << "...";
    for (int i = data_len - 8; i < data_len; ++i) {
      std::cout << cpu_data[i] << " ";
    }
    std::cout << std::endl;
    return;
  }
  int row = 1;
  int col = shape.back();
  if (shape.size() >= 2) {
    row = data_len / col;
  }
  T* cpu_data_ptr = &cpu_data.front();
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << *(cpu_data_ptr + i * col + j) << " ";
    }
    std::cout << std::endl;
  }
}
template <typename T>
void print_cpu_data(const T* data, std::vector<int> shape, std::string name) {
  int data_len = vector_prod(shape);
  std::cout << name;
  std::cout << " shape:";
  for (auto i : shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  int row = 1;
  int col = shape.back();
  if (shape.size() >= 2) {
    row = data_len / col;
  }
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << *(data + i * col + j) << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void print_vec(const std::vector<T>& data, const std::string& data_name) {
  int len = static_cast<int>(data.size());
  T sum = std::accumulate(data.begin(), data.end(), 0);
  std::cout << "DEBUG: data_name is " << data_name << ", len=" << len
            << ", sum=" << sum << ", ";
  for (int i = 0; i < len - 1; ++i) {
    std::cout << data[i] << ", ";
  }
  std::cout << data[len - 1] << std::endl;
}

#define INSTANTIATION_PRINT(T)                                           \
  template void print_vec<T>(const std::vector<T>&, const std::string&); \
  template void print_cpu_data<T>(const T*, std::vector<int>,            \
                                  std::string name);                     \
  template void print_xpu_data<T>(api::Context * ctx, const T*,          \
                                  std::vector<int>, std::string);        \
  template void print_xpu_data_all<T>(api::Context * ctx, const T*,      \
                                      std::vector<int> shape, std::string);

INSTANTIATION_PRINT(int);
INSTANTIATION_PRINT(int16_t);
INSTANTIATION_PRINT(int8_t);
INSTANTIATION_PRINT(float);
INSTANTIATION_PRINT(float16);
