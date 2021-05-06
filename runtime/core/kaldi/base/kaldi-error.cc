// base/kaldi-error.cc

// Copyright 2019 LAIX (Yi Sun)
// Copyright 2019 SmartAction LLC (kkm)
// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;  Ondrej Glembek

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-error.h"

#include <string>

namespace kaldi {

/***** GLOBAL VARIABLES FOR LOGGING *****/

int32 g_kaldi_verbose_level = 0;
static std::string program_name;  // NOLINT

void SetProgramName(const char *basename) {
  // Using the 'static std::string' for the program name is mostly harmless,
  // because (a) Kaldi logging is undefined before main(), and (b) no stdc++
  // string implementation has been found in the wild that would not be just
  // an empty string when zero-initialized but not yet constructed.
  program_name = basename;
}

}  // namespace kaldi
