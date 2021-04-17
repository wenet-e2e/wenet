// base/kaldi-error.h

// Copyright 2019 LAIX (Yi Sun)
// Copyright 2019 SmartAction LLC (kkm)
// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2009-2011  Microsoft Corporation;  Ondrej Glembek;  Lukas Burget;
//                      Saarland University

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

#ifndef KALDI_BASE_KALDI_ERROR_H_
#define KALDI_BASE_KALDI_ERROR_H_ 1

#include "utils/log.h"

namespace kaldi {

#define KALDI_WARN \
  google::LogMessage(__FILE__, __LINE__, google::WARNING).stream()
#define KALDI_ERR \
  google::LogMessage(__FILE__, __LINE__, google::ERROR).stream()
#define KALDI_LOG \
  google::LogMessage(__FILE__, __LINE__, google::INFO).stream()
#define KALDI_VLOG(v) VLOG(v)

#define KALDI_ASSERT(condition) CHECK(condition)

} // namespace kaldi

#endif // KALDI_BASE_KALDI_ERROR_H_

