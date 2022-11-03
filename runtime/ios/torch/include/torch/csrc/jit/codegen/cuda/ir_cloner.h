#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Fusion;

// Clones nodes from an exiting Fusion
class TORCH_CUDA_CU_API IrCloner : private OptInConstDispatch {
  friend class Statement;

 public:
  explicit IrCloner(Fusion* new_fusion) : fusion_(new_fusion) {}

  Statement* clone(const Statement* statement);

  template <class T>
  T* clone(const T* node) {
    return node ? clone(node->template as<Statement>())->template as<T>()
                : nullptr;
  }

  template <class T>
  std::vector<T*> clone(const std::vector<T*>& container) {
    std::vector<T*> copy;
    for (auto p : container) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      copy.push_back(clone(p));
    }
    return copy;
  }

  Fusion* fusion() const {
    return fusion_;
  }

 private:
  void registerClone(const Statement* src, Statement* clone);

  void handle(const Statement*) override;
  void handle(const Val*) override;
  void handle(const Expr*) override;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;

  void handle(const Bool*) override;
  void handle(const Float*) override;
  void handle(const Half*) override;
  void handle(const Int*) override;
  void handle(const NamedScalar*) override;

  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const TernaryOp*) override;
  void handle(const BroadcastOp*) override;
  void handle(const ReductionOp*) override;

  void handle(const Split*) override;
  void handle(const Merge*) override;

 private:
  // The destination Fusion container
  Fusion* fusion_ = nullptr;

  // The dispatch interface doesn't allow returning values from
  // individual `handle()` methods, so they are storing the
  // result here
  Statement* clone_ = nullptr;

  // We keep track of the original -> clone map so we don't
  // duplicate clones of the same object if referenced multiple times
  std::unordered_map<const Statement*, Statement*> clones_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
