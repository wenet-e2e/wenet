// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: sxc19@mails.tsinghua.edu.cn (Xingchen Song)

#ifndef BACKEND_PARAMS_H_
#define BACKEND_PARAMS_H_

#include <memory>

#include "backend/inverse_text_normalizer.h"
#include "utils/flags.h"

// InverseTextNormalizer flags
DEFINE_string(far, "", "Path to load FAR.");
DEFINE_string(rules, "ITN", "Names of the rewrite rules.");
DEFINE_string(input_mode, "byte", "Either \"byte\", \"utf8\", or the path to a "
              "symbol table for input parsing.");
DEFINE_string(output_mode, "byte", "Either \"byte\", \"utf8\", or the path to "
              "a symbol table for input parsing.");
DEFINE_string(outdir, "", "Path to export FAR. never used.");

namespace wenet {

std::shared_ptr<InverseTextNormalizer> InitInverseTextNormalizerFromFlags() {
  auto model = std::make_shared<InverseTextNormalizer>();
  if (FLAGS_far != "") {
    model->Initialize(FLAGS_far, FLAGS_rules,
                      FLAGS_input_mode, FLAGS_output_mode);
  } else {
    model = nullptr;
  }
  return model;
}

}  // namespace wenet

#endif  // BACKEND_PARAMS_H_
