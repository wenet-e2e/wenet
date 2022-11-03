#pragma once

#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor dropout(Tensor input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::dropout_(input, p, training);
  } else {
    return torch::dropout(input, p, training);
  }
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::DropoutFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout(input, F::DropoutFuncOptions().p(0.5));
/// ```
inline Tensor dropout(Tensor input,
    const DropoutFuncOptions& options = {}) {
  return detail::dropout(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor dropout2d(Tensor input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::feature_dropout_(input, p, training);
  } else {
    return torch::feature_dropout(input, p, training);
  }
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.dropout2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Dropout2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout2d(input, F::Dropout2dFuncOptions().p(0.5));
/// ```
inline Tensor dropout2d(Tensor input,
    const Dropout2dFuncOptions& options = {}) {
  return detail::dropout2d(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor dropout3d(Tensor input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::feature_dropout_(input, p, training);
  } else {
    return torch::feature_dropout(input, p, training);
  }
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.dropout3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Dropout3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::dropout3d(input, F::Dropout3dFuncOptions().p(0.5));
/// ```
inline Tensor dropout3d(Tensor input,
    const Dropout3dFuncOptions& options = {}) {
  return detail::dropout3d(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor alpha_dropout(Tensor input, double p, bool training, bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::alpha_dropout_(input, p, training) : torch::alpha_dropout(input, p, training);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.alpha_dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AlphaDropoutFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::alpha_dropout(input, F::AlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor alpha_dropout(Tensor input, const AlphaDropoutFuncOptions& options = {}) {
  return detail::alpha_dropout(input, options.p(), options.training(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor feature_alpha_dropout(Tensor input, double p, bool training, bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::feature_alpha_dropout_(input, p, training) : torch::feature_alpha_dropout(input, p, training);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.feature_alpha_dropout
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::FeatureAlphaDropoutFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::feature_alpha_dropout(input, F::FeatureAlphaDropoutFuncOptions().p(0.5).training(false));
/// ```
inline Tensor feature_alpha_dropout(Tensor input, const FeatureAlphaDropoutFuncOptions& options = {}) {
  return detail::feature_alpha_dropout(input, options.p(), options.training(), options.inplace());
}

} // namespace functional
} // namespace nn
} // namespace torch
