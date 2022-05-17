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

#include "api/wenet_api.h"

namespace py = pybind11;


PYBIND11_MODULE(_wenet, m) {
  m.doc() = "wenet pybind11 plugin";  // optional module docstring
  m.def("wenet_init", &wenet_init, py::return_value_policy::reference,
        "wenet init");
  m.def("wenet_free", &wenet_free, "wenet free");
  m.def("wenet_reset", &wenet_reset, "wenet reset");
  m.def("wenet_decode", &wenet_decode, "wenet decode");
  m.def("wenet_get_result", &wenet_get_result, py::return_value_policy::copy,
        "wenet get result");
  m.def("wenet_set_log_level", &wenet_set_log_level, "set log level");
  m.def("wenet_set_nbest", &wenet_set_nbest, "set nbest");
  m.def("wenet_set_timestamp", &wenet_set_timestamp, "set timestamp flag");
  m.def("wenet_add_context", &wenet_add_context, "add one context word");
  m.def("wenet_set_context_score", &wenet_set_context_score,
        "set context bonus score");
  m.def("wenet_set_language", &wenet_set_language, "set language");
}
