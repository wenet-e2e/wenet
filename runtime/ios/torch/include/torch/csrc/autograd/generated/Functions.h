#pragma once

// @generated from tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    result.push_back(v.unpack());
  }
  return result;
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes().vec())
    , options(t.options()) {}

  Tensor zeros() { return at::zeros(sizes, options); }

private:
  std::vector<int64_t> sizes;
  at::TensorOptions options;
};

struct TORCH_API AbsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AbsBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AcosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcosBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {


  }

  at::ScalarType other_scalar_type;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;

};
struct TORCH_API AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;

};
struct TORCH_API AddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddbmmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch2_.reset_data();
    batch2_.reset_grad_function();
    batch1_.reset_data();
    batch1_.reset_grad_function();
  }

  int64_t batch1_argsize_0 = 0;
  int64_t batch1_argsize_1 = 0;
  int64_t batch2_argsize_2 = 0;
  SavedVariable batch2_;
  at::Scalar alpha;
  SavedVariable batch1_;
  at::Scalar beta;

};
struct TORCH_API AddcdivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcdivBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor2_.reset_data();
    tensor2_.reset_grad_function();
    tensor1_.reset_data();
    tensor1_.reset_grad_function();
  }

  at::ScalarType self_scalar_type;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::Scalar value;
  SavedVariable tensor1_;
  at::ScalarType tensor2_scalar_type;

};
struct TORCH_API AddcmulBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcmulBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor2_.reset_data();
    tensor2_.reset_grad_function();
    tensor1_.reset_data();
    tensor1_.reset_grad_function();
  }

  at::ScalarType self_scalar_type;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::Scalar value;
  SavedVariable tensor1_;
  at::ScalarType tensor2_scalar_type;

};
struct TORCH_API AddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    mat2_.reset_grad_function();
    mat1_.reset_data();
    mat1_.reset_grad_function();
  }

  std::vector<int64_t> mat1_sizes;
  std::vector<int64_t> mat1_strides;
  SavedVariable mat2_;
  at::Scalar alpha;
  SavedVariable mat1_;
  std::vector<int64_t> mat2_sizes;
  std::vector<int64_t> mat2_strides;
  at::Scalar beta;

};
struct TORCH_API SparseAddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseAddmmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    sparse_.reset_data();
    sparse_.reset_grad_function();
    dense_.reset_data();
    dense_.reset_grad_function();
  }

  SavedVariable sparse_;
  std::vector<int64_t> dense_sizes;
  std::vector<int64_t> dense_strides;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable dense_;

};
struct TORCH_API AddmvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmvBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec_.reset_data();
    vec_.reset_grad_function();
    mat_.reset_data();
    mat_.reset_grad_function();
  }

  SavedVariable vec_;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat_;

};
struct TORCH_API AddrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddrBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec2_.reset_data();
    vec2_.reset_grad_function();
    vec1_.reset_data();
    vec1_.reset_grad_function();
  }

  at::Scalar beta;
  SavedVariable vec2_;
  at::Scalar alpha;
  SavedVariable vec1_;

};
struct TORCH_API AffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AffineGridGeneratorBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> size;
  bool align_corners;

};
struct TORCH_API AliasBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API AngleBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AngleBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AnyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API AnyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AllBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API AllBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AcoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AcoshBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcoshBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AsinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AsinhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinhBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanhBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API AsStridedBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward"; }
  void release_variables() override {


  }

  at::TensorGeometry self_geometry;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;

};
struct TORCH_API AsinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AtanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Atan2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Atan2Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API BaddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BaddbmmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch2_.reset_data();
    batch2_.reset_grad_function();
    batch1_.reset_data();
    batch1_.reset_grad_function();
  }

  SavedVariable batch2_;
  at::Scalar alpha;
  SavedVariable batch1_;
  at::Scalar beta;

};
struct TORCH_API BernoulliBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API BernoulliBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize p_info;

};
struct TORCH_API BernoulliBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward2"; }
  void release_variables() override {


  }



};
struct TORCH_API BmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BmmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable mat2_;

};
struct TORCH_API BmmBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BmmBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable self_;
  bool deterministic;
  SavedVariable mat2_;

};
struct TORCH_API CatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CatBackward"; }
  void release_variables() override {


  }

  std::vector<std::vector<int64_t>> tensors_args_sizes;
  std::vector<at::ScalarType> tensors_args_scalartypes;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API CauchyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CauchyBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API CeilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeilBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API CholeskyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  bool upper;
  SavedVariable result_;

};
struct TORCH_API LinalgCholeskyExBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgCholeskyExBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
    L_.reset_grad_function();
  }

  SavedVariable L_;

};
struct TORCH_API CholeskySolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskySolveBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    input2_.reset_data();
    input2_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable input2_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API CholeskyInverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyInverseBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API ClampBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    min_.reset_data();
    min_.reset_grad_function();
    max_.reset_data();
    max_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable min_;
  SavedVariable max_;

};
struct TORCH_API ClampBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> min;
  c10::optional<at::Scalar> max;

};
struct TORCH_API ClampMinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar min;

};
struct TORCH_API ClampMinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    min_.reset_data();
    min_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable min_;

};
struct TORCH_API ClampMaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar max;

};
struct TORCH_API ClampMaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    max_.reset_data();
    max_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable max_;

};
struct TORCH_API CloneBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CloneBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API CoalesceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoalesceBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API ComplexBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ComplexBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    imag_.reset_data();
    imag_.reset_grad_function();
    real_.reset_data();
    real_.reset_grad_function();
  }

  SavedVariable imag_;
  SavedVariable real_;

};
struct TORCH_API PolarBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolarBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API ConjBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConjBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API CopysignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API CopysignBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CopysignBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API CosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CosBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API CoshBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoshBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API CrossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CrossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<int64_t> dim;
  SavedVariable other_;

};
struct TORCH_API LogcumsumexpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogcumsumexpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API CumprodBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::ScalarType self_scalar_type;
  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API CumsumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  int64_t dim = 0;

};
struct TORCH_API CummaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CummaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API CumminBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumminBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API ConvTbcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvTbcBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    bias_.reset_data();
    bias_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable bias_;
  int64_t pad = 0;

};
struct TORCH_API CtcLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CtcLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    log_probs_.reset_data();
    log_probs_.reset_grad_function();
    targets_.reset_data();
    targets_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable log_probs_;
  SavedVariable targets_;
  std::vector<int64_t> input_lengths;
  std::vector<int64_t> target_lengths;
  int64_t blank = 0;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API Deg2RadBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Deg2RadBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LinalgDetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgDetBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API DiagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t diagonal = 0;

};
struct TORCH_API DiagonalBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;

};
struct TORCH_API DistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DistBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;
  at::Scalar p;
  SavedVariable result_;

};
struct TORCH_API DivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;
  at::ScalarType self_scalar_type;

};
struct TORCH_API DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;

};
struct TORCH_API DivBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;
  c10::optional<std::string> rounding_mode;
  at::ScalarType self_scalar_type;

};
struct TORCH_API DivBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward3"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;
  c10::optional<std::string> rounding_mode;

};
struct TORCH_API DotBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DotBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor_.reset_data();
    tensor_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable tensor_;
  SavedVariable self_;

};
struct TORCH_API VdotBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VdotBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FusedDropoutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedDropoutBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  double p;
  SavedVariable result1_;

};
struct TORCH_API EigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EigBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    eigenvalues_.reset_data();
    eigenvalues_.reset_grad_function();
    eigenvectors_return_.reset_data();
    eigenvectors_return_.reset_grad_function();
  }

  SavedVariable self_;
  bool eigenvectors;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_return_;

};
struct TORCH_API EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ErfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ErfcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfcBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ErfinvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfinvBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ExpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API Exp2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Exp2Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API Expm1Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Expm1Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API ExpandBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API ExponentialBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExponentialBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API FakeQuantizePerTensorAffineCachemaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineCachemaskBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API FakeQuantizeLearnablePerTensorAffineBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerTensorAffineBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    scale_.reset_data();
    scale_.reset_grad_function();
    zero_point_.reset_data();
    zero_point_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable scale_;
  SavedVariable zero_point_;
  int64_t quant_min = 0;
  int64_t quant_max = 0;
  double grad_factor;

};
struct TORCH_API FakeQuantizePerChannelAffineCachemaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerChannelAffineCachemaskBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API FakeQuantizeLearnablePerChannelAffineBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizeLearnablePerChannelAffineBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    scale_.reset_data();
    scale_.reset_grad_function();
    zero_point_.reset_data();
    zero_point_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable scale_;
  SavedVariable zero_point_;
  int64_t axis = 0;
  int64_t quant_min = 0;
  int64_t quant_max = 0;
  double grad_factor;

};
struct TORCH_API FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API FloorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FloorBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable other_;

};
struct TORCH_API FracBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FracBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API FrexpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FrexpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    exponent_.reset_grad_function();
  }

  SavedVariable exponent_;

};
struct TORCH_API GatherBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GatherBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    index_.reset_data();
    index_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable index_;
  bool sparse_grad;

};
struct TORCH_API GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GeometricBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeometricBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API GeqrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeqrfBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API GerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GerBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec2_.reset_data();
    vec2_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable vec2_;
  SavedVariable self_;

};
struct TORCH_API GridSampler2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GridSampler3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GridSampler2DCpuFallbackBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DCpuFallbackBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
  bool align_corners;

};
struct TORCH_API GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API HardsigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardsigmoidBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API HistcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HistcBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API HardswishBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardswishBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API HypotBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HypotBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    other_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable other_;
  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API I0Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "I0Backward"; }
  void release_variables() override {


  }



};
struct TORCH_API IgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammaBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API IgammacBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IgammacBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API IndexBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;

};
struct TORCH_API IndexAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexAddBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
    source_.reset_data();
    source_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;
  int64_t source_dim = 0;
  SavedVariable source_;
  at::Scalar alpha;

};
struct TORCH_API IndexCopyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexCopyBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
    source_.reset_data();
    source_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;
  int64_t source_dim = 0;
  SavedVariable source_;

};
struct TORCH_API IndexFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API IndexFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API IndexPutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexPutImplBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutImplBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexSelectBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API InverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "InverseBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API LinalgInvExBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgInvExBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    inverse_.reset_data();
    inverse_.reset_grad_function();
  }

  SavedVariable inverse_;

};
struct TORCH_API KthvalueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KthvalueBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LerpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward0"; }
  void release_variables() override {


  }

  at::Scalar weight;

};
struct TORCH_API LerpBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    weight_.reset_data();
    weight_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    end_.reset_data();
    end_.reset_grad_function();
  }

  SavedVariable weight_;
  SavedVariable self_;
  SavedVariable end_;

};
struct TORCH_API LgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LgammaBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API DigammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DigammaBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API PolygammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  int64_t n = 0;
  SavedVariable self_;

};
struct TORCH_API PolygammaBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t n = 0;

};
struct TORCH_API LogBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log10Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log10Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log1PBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log1PBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log2Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API LogaddexpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogaddexpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API Logaddexp2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Logaddexp2Backward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    other_.reset_grad_function();
  }

  at::Scalar self;
  SavedVariable other_;

};
struct TORCH_API XlogyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "XlogyBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar other;

};
struct TORCH_API SpecialXlog1PyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API SpecialXlog1PyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    other_.reset_grad_function();
  }

  at::Scalar self;
  SavedVariable other_;

};
struct TORCH_API SpecialXlog1PyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialXlog1PyBackward2"; }
  void release_variables() override {


  }

  at::Scalar other;

};
struct TORCH_API LogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogdetBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API LogNormalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogNormalBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LogsumexpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogsumexpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API LstsqBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LstsqBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LinalgLstsqBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgLstsqBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API LuWithInfoBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuWithInfoBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LuSolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuSolveBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API LuUnpackBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuUnpackBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_data_.reset_data();
    LU_data_.reset_grad_function();
  }

  SavedVariable LU_data_;
  bool unpack_data;

};
struct TORCH_API MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedScatterBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedScatterBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;
  std::vector<int64_t> source_sizes;

};
struct TORCH_API MaskedSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedSelectBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable mask_;

};
struct TORCH_API MatrixExpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MatrixExpBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MaximumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaximumBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t self_numel = 0;
  at::ScalarType self_scalar_type;

};
struct TORCH_API MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::ScalarType self_scalar_type;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API MedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API NanmedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API NanmedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanmedianBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API MinimumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinimumBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API FminBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FminBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API AmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API AminBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AminBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API MmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MmBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> mat2_sizes;
  std::vector<int64_t> mat2_strides;
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> self_strides;
  SavedVariable mat2_;

};
struct TORCH_API ModeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ModeBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
  SavedVariable other_;

};
struct TORCH_API MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar other;

};
struct TORCH_API MvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec_.reset_data();
    vec_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable vec_;
  SavedVariable self_;

};
struct TORCH_API MvlgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvlgammaBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t p = 0;

};
struct TORCH_API NanToNumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NanToNumBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API NativeBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NativeBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    grad_out_.reset_grad_function();
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_invstd_.reset_data();
    save_invstd_.reset_grad_function();
  }

  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_invstd_;
  bool train;
  double eps;

};
struct TORCH_API NativeLayerNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    bias_.reset_data();
    bias_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  std::vector<int64_t> normalized_shape;
  SavedVariable weight_;
  SavedVariable bias_;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NativeGroupNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeGroupNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  int64_t N = 0;
  int64_t C = 0;
  int64_t HxW = 0;
  int64_t group = 0;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward0"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward1"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API NegBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API NextafterBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NextafterBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API NormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar p;
  SavedVariable result_;

};
struct TORCH_API NormBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API NormBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  SavedVariable result_;

};
struct TORCH_API NormBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward3"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<at::Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API LinalgVectorNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgVectorNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar ord;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PdistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  double p;
  SavedVariable result_;

};
struct TORCH_API PdistBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackwardBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API EuclideanDistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EuclideanDistBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x1_.reset_grad_function();
    x2_.reset_data();
    x2_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;

};
struct TORCH_API CdistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x1_.reset_grad_function();
    x2_.reset_data();
    x2_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable x1_;
  SavedVariable x2_;
  double p;
  SavedVariable result_;

};
struct TORCH_API CdistBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackwardBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> mean_sizes;

};
struct TORCH_API NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward2"; }
  void release_variables() override {


  }

  std::vector<int64_t> std_sizes;

};
struct TORCH_API NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward3"; }
  void release_variables() override {


  }

  std::vector<int64_t> mean_sizes;
  std::vector<int64_t> std_sizes;

};
struct TORCH_API LinalgHouseholderProductBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgHouseholderProductBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    tau_.reset_data();
    tau_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable tau_;

};
struct TORCH_API OrmqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrmqrBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API PermuteBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> dims;

};
struct TORCH_API PoissonBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PoissonBackward"; }
  void release_variables() override {


  }

  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API PowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar exponent;

};
struct TORCH_API PowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    exponent_.reset_data();
    exponent_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable exponent_;
  SavedVariable result_;

};
struct TORCH_API PowBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward2"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    exponent_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar self;
  SavedVariable exponent_;
  SavedVariable result_;

};
struct TORCH_API ProdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API ProdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PutBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
    source_.reset_data();
    source_.reset_grad_function();
  }

  SavedVariable index_;
  torch::autograd::generated::TypeAndSize source_info;
  bool accumulate;
  SavedVariable source_;

};
struct TORCH_API LinalgQrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgQrBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    Q_.reset_data();
    Q_.reset_grad_function();
    R_.reset_data();
    R_.reset_grad_function();
  }

  SavedVariable self_;
  std::string mode;
  SavedVariable Q_;
  SavedVariable R_;

};
struct TORCH_API Rad2DegBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rad2DegBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward2"; }
  void release_variables() override {


  }



};
struct TORCH_API ReciprocalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReciprocalBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward0"; }
  void release_variables() override {


  }



};
struct TORCH_API RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward1"; }
  void release_variables() override {


  }



};
struct TORCH_API RenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RenormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar p;
  int64_t dim = 0;
  at::Scalar maxnorm;

};
struct TORCH_API RepeatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RepeatBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> repeats;

};
struct TORCH_API SpecialEntrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SpecialEntrBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API RoundBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API RsqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsqrtBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API ScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API ScatterBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API ScatterAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterAddBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct TORCH_API SelectBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  int64_t index = 0;

};
struct TORCH_API SigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API LogitBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogitBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<double> eps;

};
struct TORCH_API SignBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SignBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API SgnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SgnBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API SinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API SincBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SincBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API SinhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinhBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API SliceBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  c10::optional<int64_t> start;
  c10::optional<int64_t> end;
  int64_t step = 0;

};
struct TORCH_API SlogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlogdetBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    sign_.reset_data();
    sign_.reset_grad_function();
    logabsdet_.reset_data();
    logabsdet_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable sign_;
  SavedVariable logabsdet_;

};
struct TORCH_API LinalgSlogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSlogdetBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    sign_.reset_data();
    sign_.reset_grad_function();
    logabsdet_.reset_data();
    logabsdet_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable sign_;
  SavedVariable logabsdet_;

};
struct TORCH_API SolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SolveBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    solution_.reset_data();
    solution_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  SavedVariable solution_;

};
struct TORCH_API LinalgSolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgSolveBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable other_;
  SavedVariable result_;

};
struct TORCH_API SortBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API SortBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API SplitBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  int64_t split_size = 0;
  int64_t dim = 0;

};
struct TORCH_API UnsafeSplitBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  int64_t split_size = 0;
  int64_t dim = 0;

};
struct TORCH_API SplitWithSizesBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<int64_t> split_sizes;
  int64_t dim = 0;

};
struct TORCH_API UnsafeSplitWithSizesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeSplitWithSizesBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  at::TensorOptions self_options;
  std::vector<int64_t> split_sizes;
  int64_t dim = 0;

};
struct TORCH_API SqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqrtBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API SqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward1 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API SqueezeBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward3"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API StdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API StdMeanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdMeanBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward0"; }
  void release_variables() override {


  }

  at::ScalarType other_scalar_type;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;

};
struct TORCH_API SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;

};
struct TORCH_API RsubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward0"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::ScalarType other_scalar_type;
  at::Scalar alpha;

};
struct TORCH_API RsubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward1"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;
  at::Scalar alpha;

};
struct TORCH_API SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API NansumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NansumBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  at::ScalarType self_scalar_type;
  SavedVariable self_;

};
struct TORCH_API NansumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NansumBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  at::ScalarType self_scalar_type;
  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API SvdHelperBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SvdHelperBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    U_.reset_data();
    U_.reset_grad_function();
    S_.reset_data();
    S_.reset_grad_function();
    V_.reset_data();
    V_.reset_grad_function();
  }

  SavedVariable self_;
  bool some;
  bool compute_uv;
  SavedVariable U_;
  SavedVariable S_;
  SavedVariable V_;

};
struct TORCH_API SymeigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SymeigBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    eigenvalues_.reset_data();
    eigenvalues_.reset_grad_function();
    eigenvectors_return_.reset_data();
    eigenvectors_return_.reset_grad_function();
  }

  SavedVariable self_;
  bool eigenvectors;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_return_;

};
struct TORCH_API LinalgEighBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEighBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    eigenvalues_.reset_data();
    eigenvalues_.reset_grad_function();
    eigenvectors_.reset_data();
    eigenvectors_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
struct TORCH_API LinalgEigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LinalgEigBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    eigenvalues_.reset_data();
    eigenvalues_.reset_grad_function();
    eigenvectors_.reset_data();
    eigenvectors_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;

};
struct TORCH_API TBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API FlipBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlipBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> dims;

};
struct TORCH_API RollBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RollBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> shifts;
  std::vector<int64_t> dims;

};
struct TORCH_API Rot90Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rot90Backward"; }
  void release_variables() override {


  }

  int64_t k = 0;
  std::vector<int64_t> dims;

};
struct TORCH_API TakeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TakeBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    index_.reset_grad_function();
  }

  torch::autograd::generated::TypeAndSize self_info;
  SavedVariable index_;

};
struct TORCH_API TanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API TanhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API TopkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TopkBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API TraceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TraceBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API TransposeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0"; }
  void release_variables() override {


  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TransposeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward1"; }
  void release_variables() override {


  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TriangularSolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriangularSolveBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    solution_.reset_data();
    solution_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  bool upper;
  bool transpose;
  bool unitriangular;
  SavedVariable solution_;

};
struct TORCH_API TrilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilBackward"; }
  void release_variables() override {


  }

  int64_t diagonal = 0;

};
struct TORCH_API TriuBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriuBackward"; }
  void release_variables() override {


  }

  int64_t diagonal = 0;

};
struct TORCH_API TruncBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TruncBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API ToDenseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToDenseBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ToSparseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API ToMkldnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToMkldnnBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API UnfoldBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  int64_t dimension = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct TORCH_API UnfoldBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackwardBackward"; }
  void release_variables() override {


  }

  int64_t dim = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct TORCH_API UniformBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniformBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueDimBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueConsecutiveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueConsecutiveBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API UniqueDimConsecutiveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueDimConsecutiveBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API Unique2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Unique2Backward"; }
  void release_variables() override {


  }



};
struct TORCH_API UnsafeViewBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeViewBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API UnsqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API UnsqueezeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward1"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API VarBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;

};
struct TORCH_API VarMeanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarMeanBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  c10::OptionalArray<int64_t> dim;
  c10::optional<int64_t> correction;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API ViewBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API ViewAsRealBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsRealBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API ViewAsComplexBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewAsComplexBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API SWhereBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SWhereBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    condition_.reset_data();
    condition_.reset_grad_function();
  }

  SavedVariable condition_;

};
struct TORCH_API WeightNormCudaInterfaceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "WeightNormCudaInterfaceBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    v_.reset_data();
    v_.reset_grad_function();
    g_.reset_data();
    g_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable v_;
  SavedVariable g_;
  int64_t dim = 0;
  SavedVariable result1_;

};
struct TORCH_API ZeroBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ZeroBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API SparseMaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMaskBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API SparseCooTensorWithDimsAndTensorsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseCooTensorWithDimsAndTensorsBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;

};
struct TORCH_API SparseSumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSumBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;

};
struct TORCH_API StandardGammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API StandardGammaGradBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaGradBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API ValuesBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct TORCH_API TrilinearBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilinearBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    i1_.reset_data();
    i1_.reset_grad_function();
    i2_.reset_data();
    i2_.reset_grad_function();
    i3_.reset_data();
    i3_.reset_grad_function();
  }

  SavedVariable i1_;
  SavedVariable i2_;
  SavedVariable i3_;
  std::vector<int64_t> expand1;
  std::vector<int64_t> expand2;
  std::vector<int64_t> expand3;
  std::vector<int64_t> sumdim;
  int64_t unroll_dim = 0;

};
struct TORCH_API ConstantPadNdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConstantPadNdBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> pad;

};
struct TORCH_API BinaryCrossEntropyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct TORCH_API BinaryCrossEntropyBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  SavedVariable grad_output_;

};
struct TORCH_API BinaryCrossEntropyWithLogitsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyWithLogitsBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    pos_weight_.reset_data();
    pos_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable pos_weight_;
  int64_t reduction = 0;

};
struct TORCH_API EmbeddingBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  int64_t weight_argsize_0 = 0;
  SavedVariable indices_;
  int64_t padding_idx = 0;
  bool scale_grad_by_freq;
  bool sparse;

};
struct TORCH_API EmbeddingDenseBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingDenseBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  int64_t padding_idx = 0;

};
struct TORCH_API EmbeddingBagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    weight_.reset_data();
    weight_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
    offsets_.reset_data();
    offsets_.reset_grad_function();
    per_sample_weights_.reset_data();
    per_sample_weights_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
  }

  SavedVariable weight_;
  SavedVariable indices_;
  SavedVariable offsets_;
  int64_t mode = 0;
  int64_t padding_idx = 0;
  int64_t weight_argsize_0 = 0;
  bool scale_grad_by_freq;
  bool sparse;
  SavedVariable per_sample_weights_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct TORCH_API EmbeddingRenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingRenormBackward"; }
  void release_variables() override {


  }



};
struct TORCH_API KlDivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  bool log_target;

};
struct TORCH_API L1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API MseLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API MultiMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultiMarginLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  at::Scalar p;
  at::Scalar margin;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct TORCH_API MultilabelMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultilabelMarginLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    is_target_.reset_data();
    is_target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  SavedVariable is_target_;

};
struct TORCH_API NllLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    total_weight_.reset_data();
    total_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct TORCH_API NllLoss2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    total_weight_.reset_data();
    total_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct TORCH_API SmoothL1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double beta;

};
struct TORCH_API HuberLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double delta;

};
struct TORCH_API SoftMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API ReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API SiluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SiluBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API MishBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MishBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API EluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;

};
struct TORCH_API EluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;
  SavedVariable result_;

};
struct TORCH_API CeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar alpha;

};
struct TORCH_API CeluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar alpha;
  SavedVariable result_;

};
struct TORCH_API GeluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeluBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API GluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;

};
struct TORCH_API HardshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API HardshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API HardtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar min_val;
  at::Scalar max_val;

};
struct TORCH_API HardtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar min_val;
  at::Scalar max_val;
  SavedVariable result_;

};
struct TORCH_API LeakyReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar negative_slope;

};
struct TORCH_API LeakyReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar negative_slope;
  SavedVariable result_;

};
struct TORCH_API LogSigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    buffer_.reset_data();
    buffer_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable buffer_;

};
struct TORCH_API LogSoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseLogSoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseLogSoftmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API PreluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API PreluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API RreluWithNoiseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    noise_.reset_data();
    noise_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;

};
struct TORCH_API RreluWithNoiseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    noise_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;
  SavedVariable result_;

};
struct TORCH_API SoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseSoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSoftmaxBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct TORCH_API SparseSparseMatmulBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSparseMatmulBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct TORCH_API SoftplusBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar beta;
  at::Scalar threshold;
  SavedVariable result_;

};
struct TORCH_API SoftshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API ThresholdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar threshold;

};
struct TORCH_API ThresholdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward1"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
    result_.reset_grad_function();
  }

  at::Scalar threshold;
  SavedVariable result_;

};
struct TORCH_API ReflectionPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReflectionPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API ReplicationPad3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct TORCH_API UpsampleLinear1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleBilinear2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleBicubic2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleTrilinear3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleNearest2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleLinear1DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBilinear2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleTrilinear3DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBicubic2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest1DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest2DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest3DBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward1"; }
  void release_variables() override {


  }

  std::vector<int64_t> input_sizes;
  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API AdaptiveAvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveAvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AdaptiveMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API AvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API FractionalMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API FractionalMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API MaxPool2DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxPool3DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxUnpool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;

};
struct TORCH_API MaxUnpool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API ConvolutionOverrideableBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionOverrideableBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API ConvolutionBackwardOverrideableBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackwardOverrideableBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API SlowConvTranspose2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ThnnConv2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    finput_.reset_data();
    finput_.reset_grad_function();
    fgrad_input_.reset_data();
    fgrad_input_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct TORCH_API ThnnConv2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API ThnnConvDepthwise2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDepthwise2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ThnnConvDepthwise2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDepthwise2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  int64_t self_argsize_1 = 0;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ConvDepthwise3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API ConvDepthwise3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvDepthwise3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  int64_t self_argsize_1 = 0;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API SlowConv3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    finput_.reset_data();
    finput_.reset_grad_function();
    fgrad_input_.reset_data();
    fgrad_input_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct TORCH_API SlowConv3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConv3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct TORCH_API SlowConvDilated2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvDilated3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API Col2ImBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Im2ColBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackward"; }
  void release_variables() override {


  }

  int64_t self_argsize_2 = 0;
  int64_t self_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Im2ColBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Col2ImBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackwardBackward"; }
  void release_variables() override {


  }

  int64_t grad_output_argsize_2 = 0;
  int64_t grad_output_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API AdaptiveAvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable grad_output_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveAvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable grad_output_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AdaptiveMaxPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API AvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API EluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_or_result_.reset_data();
    self_or_result_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  at::Scalar alpha;
  at::Scalar scale;
  at::Scalar input_scale;
  bool is_result;
  SavedVariable self_or_result_;
  SavedVariable grad_output_;

};
struct TORCH_API FractionalMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API FractionalMaxPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API GluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable grad_output_;

};
struct TORCH_API HardtanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar min_val;
  at::Scalar max_val;

};
struct TORCH_API KlDivBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  bool log_target;

};
struct TORCH_API L1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API LogSigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    buffer_.reset_data();
    buffer_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable buffer_;
  SavedVariable grad_output_;

};
struct TORCH_API LogSoftmaxBackwardDataBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackwardDataBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable self_;

};
struct TORCH_API LeakyReluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar negative_slope;

};
struct TORCH_API MaxPool2DWithIndicesBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MaxPool3DWithIndicesBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MaxUnpool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  std::vector<int64_t> output_size;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API MseLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API NllLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct TORCH_API NllLoss2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct TORCH_API RreluWithNoiseBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    noise_.reset_data();
    noise_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable noise_;
  at::Scalar lower;
  at::Scalar upper;
  bool training;

};
struct TORCH_API ReflectionPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReflectionPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API ReplicationPad3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackwardBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;

};
struct TORCH_API SmoothL1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double beta;

};
struct TORCH_API HuberLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HuberLossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  double delta;

};
struct TORCH_API SoftplusBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar beta;
  at::Scalar threshold;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API SoftmaxBackwardDataBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackwardDataBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable grad_output_;

};
struct TORCH_API SoftMarginLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API SoftshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar lambd;

};
struct TORCH_API ThresholdBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  at::Scalar threshold;

};
struct TORCH_API UpsampleLinear1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleBilinear2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleBicubic2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleTrilinear3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  bool align_corners;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales;

};
struct TORCH_API UpsampleNearest2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleNearest3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward0"; }
  void release_variables() override {


  }

  std::vector<int64_t> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;

};
struct TORCH_API UpsampleLinear1DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBilinear2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleTrilinear3DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleBicubic2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  bool align_corners;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest1DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest2DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API UpsampleNearest3DBackwardBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward1"; }
  void release_variables() override {


  }

  c10::OptionalArray<int64_t> output_size;
  c10::OptionalArray<double> scale_factors;

};
struct TORCH_API SigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API TanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TORCH_API CudnnCtcLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API CudnnConvolutionTransposeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionTransposeBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

};
struct TORCH_API CudnnGridSamplerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnGridSamplerBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grid_;

};
struct TORCH_API CudnnAffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnAffineGridGeneratorBackward"; }
  void release_variables() override {


  }

  int64_t N = 0;
  int64_t C = 0;
  int64_t H = 0;
  int64_t W = 0;

};
struct TORCH_API CudnnBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct TORCH_API CudnnBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_var_.reset_data();
    save_var_.reset_grad_function();
    reserveSpace_.reset_data();
    reserveSpace_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;
  SavedVariable reserveSpace_;

};
struct TORCH_API NnpackSpatialConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NnpackSpatialConvolutionBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable input_;
  int64_t weight_argsize_2 = 0;
  int64_t weight_argsize_3 = 0;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API CudnnRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.clear();
    weight_released_ = true;
    hx_.reset_data();
    hx_.reset_grad_function();
    cx_.reset_data();
    cx_.reset_grad_function();
    dropout_state_.reset_data();
    dropout_state_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
    result4_.reset_data();
    result4_.reset_grad_function();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode = 0;
  int64_t hidden_size = 0;
  int64_t proj_size = 0;
  int64_t num_layers = 0;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct TORCH_API CudnnRnnBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackwardBackward"; }
  void release_variables() override {


  }


  size_t weight_size_;
};
struct TORCH_API MiopenConvolutionTransposeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionTransposeBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenDepthwiseConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenDepthwiseConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct TORCH_API MiopenBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API MiopenBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_var_.reset_data();
    save_var_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;

};
struct TORCH_API MiopenRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenRnnBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    input_.reset_grad_function();
    weight_.clear();
    weight_released_ = true;
    hx_.reset_data();
    hx_.reset_grad_function();
    cx_.reset_data();
    cx_.reset_grad_function();
    dropout_state_.reset_data();
    dropout_state_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
    result4_.reset_data();
    result4_.reset_grad_function();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode = 0;
  int64_t hidden_size = 0;
  int64_t num_layers = 0;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct TORCH_API MkldnnConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API MkldnnConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackwardBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API MkldnnLinearBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnLinearBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;

};
struct TORCH_API MkldnnMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result_;

};
struct TORCH_API MkldnnMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnMaxPool3DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result_;

};
struct TORCH_API MkldnnAdaptiveAvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnAdaptiveAvgPool2DBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API MkldnnReshapeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnReshapeBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API FftR2CBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftR2CBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool onesided;

};
struct TORCH_API FftC2RBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2RBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> dim;
  int64_t normalization = 0;

};
struct TORCH_API FftC2CBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftC2CBackward"; }
  void release_variables() override {


  }

  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool forward;

};
struct TORCH_API UnbindBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward"; }
  void release_variables() override {


  }

  int64_t dim = 0;

};
struct TORCH_API StackBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StackBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensors_.clear();
    tensors_released_ = true;
  }

  std::vector<SavedVariable> tensors_;
  bool tensors_released_ = false;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API ThnnFusedLstmCellBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedLstmCellBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_gates_.reset_data();
    input_gates_.reset_grad_function();
    hidden_gates_.reset_data();
    hidden_gates_.reset_grad_function();
    cx_.reset_data();
    cx_.reset_grad_function();
    input_bias_.reset_data();
    input_bias_.reset_grad_function();
    hidden_bias_.reset_data();
    hidden_bias_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_gates_;
  SavedVariable hidden_gates_;
  SavedVariable cx_;
  SavedVariable input_bias_;
  SavedVariable hidden_bias_;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API ThnnFusedGruCellBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedGruCellBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_gates_.reset_data();
    input_gates_.reset_grad_function();
    hidden_gates_.reset_data();
    hidden_gates_.reset_grad_function();
    hx_.reset_data();
    hx_.reset_grad_function();
    input_bias_.reset_data();
    input_bias_.reset_grad_function();
    hidden_bias_.reset_data();
    hidden_bias_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable input_gates_;
  SavedVariable hidden_gates_;
  SavedVariable hx_;
  SavedVariable input_bias_;
  SavedVariable hidden_bias_;
  SavedVariable result1_;

};
struct TORCH_API PackPaddedSequenceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PackPaddedSequenceBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> input_sizes;
  bool batch_first;
  SavedVariable result1_;

};
struct TORCH_API SegmentReduceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SegmentReduceBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.reset_data();
    data_.reset_grad_function();
    lengths_.reset_data();
    lengths_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable data_;
  SavedVariable lengths_;
  SavedVariable result_;

};

}}} // namespace torch::autograd::generated
