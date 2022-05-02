// Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
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

#include <pybind11/pybind11.h>

namespace py = pybind11;

void* wenet_init(const char* model_dir) { return nullptr; }
void wenet_free(void* decoder) {}
void wenet_reset(void* decoder) {}
void wenet_decode(void* decoder,
                  const char* data,
                  int len,
                  int last = 1) {}
const char* wenet_get_result(void* decoder) {return nullptr; }



PYBIND11_MODULE(_wenet, m) {
  m.doc() = "wenet pybind11 plugin";  // optional module docstring
  m.def("wenet_init", &wenet_init, py::return_value_policy::reference,
        "wenet init");
  m.def("wenet_free", &wenet_free,
        "wenet free");
  m.def("wenet_reset", &wenet_reset,
        "wenet reset");
  m.def("wenet_decode", &wenet_decode,
        "wenet decode");
  m.def("wenet_get_result", &wenet_get_result, py::return_value_policy::copy,
        "wenet get result");
}
