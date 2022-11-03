#pragma once

#include <torch/nn/options/batchnorm.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor batch_norm(const Tensor& input,
                         const Tensor& running_mean,
                         const Tensor& running_var,
                         Tensor weight,
                         Tensor bias,
                         bool training,
                         c10::optional<double> momentum,
                         double eps) {
  if (training) {
    auto size = input.sizes();
    int64_t size_prods = size[0];
    for (size_t i = 0; i < size.size() - 2; i++) {
      size_prods *= size[i + 2];
    }
    TORCH_CHECK(size_prods != 1,
                "Expected more than 1 value per channel when training, got input size ", size);
  }

  return torch::batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum.value(),
    eps,
    at::globalContext().userEnabledCuDNN());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.batch_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::BatchNormFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::batch_norm(input, mean, variance, F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(0.1).eps(1e-05).training(false));
/// ```
inline Tensor batch_norm(const Tensor& input, const Tensor& running_mean,
                         const Tensor& running_var, const BatchNormFuncOptions& options = {}) {
  return detail::batch_norm(
    input,
    running_mean,
    running_var,
    options.weight(),
    options.bias(),
    options.training(),
    options.momentum(),
    options.eps());
}

} // namespace functional
} // namespace nn
} // namespace torch
