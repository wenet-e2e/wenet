// base/kaldi-error.h

// Copyright (c) 2021 Mobvoi Inc (Binbin Zhang)
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

#ifndef KALDI_BASE_KALDI_ERROR_H_
#define KALDI_BASE_KALDI_ERROR_H_ 1

#include "utils/log.h"

namespace kaldi {

#define KALDI_WARN \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_WARNING).stream()
#define KALDI_ERR \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_ERROR).stream()
#define KALDI_LOG \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream()
#define KALDI_VLOG(v) VLOG(v)

#define KALDI_ASSERT(condition) CHECK(condition)


/***** PROGRAM NAME AND VERBOSITY LEVEL *****/

/// Called by ParseOptions to set base name (no directory) of the executing
/// program. The name is printed in logging code along with every message,
/// because in our scripts, we often mix together the stderr of many programs.
/// This function is very thread-unsafe.
void SetProgramName(const char *basename);

/// This is set by util/parse-options.{h,cc} if you set --verbose=? option.
/// Do not use directly, prefer {Get,Set}VerboseLevel().
extern int32 g_kaldi_verbose_level;

/// Get verbosity level, usually set via command line '--verbose=' switch.
inline int32 GetVerboseLevel() { return g_kaldi_verbose_level; }

/// This should be rarely used, except by programs using Kaldi as library;
/// command-line programs set the verbose level automatically from ParseOptions.
inline void SetVerboseLevel(int32 i) { g_kaldi_verbose_level = i; }

}  // namespace kaldi

#endif  // KALDI_BASE_KALDI_ERROR_H_

