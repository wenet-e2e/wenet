#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <torch/enum.h>

namespace torch {
namespace nn {

/// Options for the `TransformerEncoderLayer`
///
/// Example:
/// ```
/// auto options = TransformerEncoderLayer(512, 8).dropout(0.2);
/// ```
struct TORCH_API TransformerEncoderLayerOptions {

  using activation_t = c10::variant<enumtype::kReLU, enumtype::kGELU>;

  /* implicit */ TransformerEncoderLayerOptions(int64_t d_model, int64_t nhead);

  /// the number of expected features in the input
  TORCH_ARG(int64_t, d_model);

  /// the number of heads in the multiheadattention models
  TORCH_ARG(int64_t, nhead);

  /// the dimension of the feedforward network model, default is 2048
  TORCH_ARG(int64_t, dim_feedforward) = 2048;

  /// the dropout value, default is 0.1
  TORCH_ARG(double, dropout) = 0.1;

  /// the activation function of intermediate layer, either ``torch::kReLU`` or ``torch::GELU``, default is ``torch::kReLU``
  TORCH_ARG(activation_t, activation) = torch::kReLU;
};


// ============================================================================

/// Options for the `TransformerDecoderLayer` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512, 8).dropout(0.2));
/// ```
struct TORCH_API TransformerDecoderLayerOptions {

  using activation_t = c10::variant<enumtype::kReLU, enumtype::kGELU>;

  TransformerDecoderLayerOptions(int64_t d_model, int64_t nhead);

  /// number of expected features in the input
  TORCH_ARG(int64_t, d_model);

  /// number of heads in the multiheadattention models
  TORCH_ARG(int64_t, nhead);

  /// dimension of the feedforward network model. Default: 2048
  TORCH_ARG(int64_t, dim_feedforward) = 2048;

  /// dropout value. Default: 1
  TORCH_ARG(double, dropout) = 0.1;

  /// activation function of intermediate layer, can be either ``torch::kGELU`` or ``torch::kReLU``. Default: ``torch::kReLU``
  TORCH_ARG(activation_t, activation) = torch::kReLU;
};


} // namespace nn
} // namespace torch
