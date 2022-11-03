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

#include <chrono>
#include <mutex>
#include <thread>
#include <tuple>
#include "xpu_conformer.h"  // NOLINT
#include "xpu_util.h"       // NOLINT
namespace api = baidu::xpu::api;
namespace wenet = xpu::wenet;

template <typename T, typename TW, typename TGEMM>
static void conformer_test(const std::string& data_dir,
                           const std::string& params_dir, int threads_number,
                           int dev_id) {
  typedef std::vector<
      std::tuple<std::tuple<float*, std::vector<int>>,
                 std::tuple<std::vector<int>, std::vector<int>>>>
      Dtype;
  ConformerEncoderParam<T, TW> encoder_param;
  init_encoder_params<T, TW>(params_dir, encoder_param);
  ConformerDecoderParam<T, TW> decoder_param;
  init_decoder_params<T, TW>(params_dir, decoder_param);
  int real_threads_number = threads_number <= 0 ? 1 : threads_number;
  std::cout << "Encoder + Decoder MultiStreamTest threads:"
            << real_threads_number << std::endl;
  // init test data
  std::vector<int> ids = get_all_ids(data_dir);
  Dtype data_list;
  for (auto index_id : ids) {
    std::string input_lenghts_prefix =
        data_dir + std::to_string(index_id) + "_len";
    std::string input_prefix = data_dir + std::to_string(index_id);
    auto input_lenghts_cpu_info =
        read_cpu_data_from_file<int>(input_lenghts_prefix, 1);
    auto input_xpu_info = read_xpu_data_from_file<float>(input_prefix, 3);
    data_list.push_back(
        std::make_tuple(input_xpu_info, input_lenghts_cpu_info));
  }
  bool write_res = true;
  // init mem
  int ret = 0;
  std::vector<api::Context*> ctx_xpu_ptrs(real_threads_number);
  std::vector<XPUStream> streams(real_threads_number);

  int nsdnn = real_threads_number > 1 ? 2 : 6;
  int ncluster = real_threads_number > 1 ? 2 : 8;
  for (int i = 0; i < real_threads_number; i++) {
    ret = xpu_stream_create(&streams[i]);
    ctx_xpu_ptrs[i] = new api::Context(api::kXPU2);
    ctx_xpu_ptrs[i]->xpu_stream = streams[i];
    ctx_xpu_ptrs[i]->set_nsdnn(nsdnn);
    ctx_xpu_ptrs[i]->set_ncluster(ncluster);
  }
  // threads
  std::vector<float> thread_times(real_threads_number);
  std::vector<std::thread> threads;
  int data_counter = 0;
  std::mutex data_mutex;
  std::vector<float> time_info(real_threads_number, 0.0f);
  auto f = [&](int thread_id) {
    xpu_set_device(dev_id);
    api::Context* ctx_xpu = ctx_xpu_ptrs[thread_id];
    api::ctx_guard RAII_GUARD(ctx_xpu);
    while (true) {
      int data_index = -1;
      data_mutex.lock();
      if (data_counter >= data_list.size()) {
        data_mutex.unlock();
        break;
      }
      data_index = data_counter++;
      data_mutex.unlock();
      if (data_index < 0) {
        continue;
      }
      auto start_time = std::chrono::system_clock::now();
      // get input data
      auto& input_xpu_info = std::get<0>(data_list[data_index]);
      auto& input_lenghts_info = std::get<1>(data_list[data_index]);
      auto& input_xpu_data = std::get<0>(input_xpu_info);
      auto& speech_shape = std::get<1>(input_xpu_info);
      int batch = speech_shape[0];
      int max_seqlen = speech_shape[1];
      auto xpu_mask_info_float = create_mask_according_speech_length<float>(
          std::get<0>(input_lenghts_info), max_seqlen, ctx_xpu->xpu_stream);
      ret = xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      int q_seqlen = ((max_seqlen - 1) / 2 - 1) / 2;
      // encoder run
      int att_dim = encoder_param.head_num * encoder_param.head_dim;
      int ctc_dim = encoder_param.ctc_dim;
      T* encoder_out = RAII_GUARD.alloc<T>(batch * q_seqlen * att_dim);
      T* ctc_probs = RAII_GUARD.alloc<T>(batch * q_seqlen * ctc_dim);
      // get encoder_out & ctc_probs
      ret = wenet::conformer_encoder_wenet<T, TW, TGEMM>(
          ctx_xpu, input_xpu_data, speech_shape, encoder_out, ctc_probs,
          encoder_param, xpu_mask_info_float);
      CHECK_RET(ret);
      ret = xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      // ctc_prefix_beamsearch implement in cpu
      int beam_size = encoder_param.beam_size;
      int new_bs = batch * beam_size;
      std::vector<int> hyps_len(new_bs);
      std::vector<float> ctc_scores(new_bs);
      std::vector<int> hyps_cpu;
      int* hyps = RAII_GUARD.alloc<int>(new_bs * q_seqlen);
      ret = wenet::ctc_prefix_beamsearch<T>(ctx_xpu, ctc_probs, hyps_cpu,
                                            hyps_len, ctc_scores, batch,
                                            beam_size, q_seqlen, ctc_dim);
      CHECK_RET(ret);
      ret = xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      int max_target_len =
          padding_target(hyps_cpu, hyps_len, beam_size, decoder_param.eos_id);
      ret = xpu_memcpy(hyps, reinterpret_cast<void*>(&hyps_cpu[0]),
                       max_target_len * new_bs * sizeof(int),
                       XPUMemcpyKind::XPU_HOST_TO_DEVICE);
      ret = xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      // decoder
      int pad_target_len = decoder_param.add_sos_num + max_target_len;
      float* character_scores =
          RAII_GUARD.alloc<float>(new_bs * pad_target_len * ctc_dim);
      ret = wenet::conformer_decoder_wenet<T, TW, TGEMM>(
          ctx_xpu, encoder_out, {batch, q_seqlen, att_dim},
          std::get<0>(xpu_mask_info_float), hyps, {new_bs, max_target_len},
          character_scores, decoder_param);
      CHECK_RET(ret);
      ret = xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      // Only use decoder score for rescoring
      std::vector<float> best_score(batch, -std::numeric_limits<float>::max());
      std::vector<int> best_index(batch, 0);
      float ctc_weight = 0.5;
      std::vector<float> decoder_out(new_bs * pad_target_len * ctc_dim);
      ret = xpu_memcpy(&decoder_out[0], character_scores,
                       new_bs * max_target_len * ctc_dim * sizeof(float),
                       XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      xpu_wait(ctx_xpu->xpu_stream);
      CHECK_RET(ret);
      // cal score && output
      std::string wav_prefix =
          data_dir + std::to_string(data_index) + "_wav.txt";
      std::string res_prefix = "./token_id.txt";
      std::ofstream res;
      std::string wav_name;
      std::vector<std::string> wav_info;
      if (write_res) {
        std::ifstream wav(wav_prefix.c_str());
        if (!wav.is_open()) {
          std::cout << "wav file open fail" << std::endl;
          exit(0);
        }
        while (getline(wav, wav_name)) {
          wav_info.push_back(wav_name);
        }
        wav.close();
      }
      for (int i = 0; i < batch; i++) {
        for (int j = 0; j < beam_size; j++) {
          T score = 0.0;
          for (int k = 0; k < hyps_len[i * beam_size + j]; k++) {
            int index = i * beam_size * max_target_len * ctc_dim +
                        j * max_target_len * ctc_dim + k * ctc_dim +
                        hyps_cpu[k];
            score += decoder_out[index];
          }
          score += decoder_out[i * beam_size * max_target_len * ctc_dim +
                               j * max_target_len * ctc_dim +
                               hyps_len[i * batch + j] * ctc_dim + ctc_dim - 1];
          // add ctc score
          score += ctc_weight * ctc_scores[i * beam_size + j];
          if (score > best_score[i]) {
            best_score[i] = score;
            best_index[i] = j;
          }
        }
        int token_index = best_index[i] + i * beam_size;
        if (write_res) {
          data_mutex.lock();
          res.open(res_prefix, std::ios::app);
          if (!res.is_open()) {
            std::cout << "res file open fail" << std::endl;
            exit(0);
          }
          res << wav_info[i] << ":";
          for (int k = 0; k < hyps_len[token_index]; k++)
            res << hyps_cpu[k] << " ";
          res << std::endl;
          res.close();
          data_mutex.unlock();
        }
      }
      auto end_time = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      time_info[thread_id] += static_cast<float>(duration.count()) / 1000;
      ret = xpu_free(std::get<0>(input_xpu_info));
      CHECK_RET(ret);
      ret = xpu_free(std::get<0>(xpu_mask_info_float));
      CHECK_RET(ret);
    }
  };
  auto all_start = std::chrono::system_clock::now();
  for (auto i = 0; i < real_threads_number; i++) {
    std::thread t(f, i);
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
  auto all_end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      all_end - all_start);
  float total_time = static_cast<float>(duration.count()) / 1000;
  std::cout << "Total time cost:" << total_time << std::endl;
  for (int i = 0; i < real_threads_number; i++) {
    if (ctx_xpu_ptrs[i]) delete ctx_xpu_ptrs[i];
  }
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Only support the following three params:" << std::endl;
    std::cout
        << "\t1. " << argv[0]
        << " encoder_test [params_dir] [data_dir] [dev_id] [threads_number]"
        << std::endl;
    std::cout
        << "\t2. " << argv[0]
        << " decoder_test [params_dir] [data_dir] [dev_id] [threads_number]"
        << std::endl;
    std::cout << "\t3. " << argv[0]
              << " all [params_dir] [data_dir] [dev_id] [threads_number]"
              << std::endl;
    return 0;
  }
  std::string mode = argv[1];
  std::string params_dir = argv[2];
  std::string data_dir = argv[3];
  int dev_id = std::stoi(argv[4]);
  int threads_number = std::stoi(argv[5]);
  add_separator_when_necessary(params_dir);
  add_separator_when_necessary(data_dir);
  xpu_set_device(dev_id);

  typedef float16 T;
  typedef int16_t TW;
  typedef int16_t TGEMM;

  if (mode == "all") {
    conformer_test<T, TW, TGEMM>(data_dir, params_dir, threads_number, dev_id);
  } else {
    std::cout << "Unkown test mode: " << mode << std::endl;
    std::exit(1);
  }
}
