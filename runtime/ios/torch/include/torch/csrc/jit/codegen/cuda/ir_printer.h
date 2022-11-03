#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Prints computation Fusion IR nodes
//!
//! IrMathPrinter and IrTransformPrinter allow the splitting up of fusion print
//! functions. IrMathPrinter as its name implies focuses solely on what tensor
//! computations are taking place. Resulting TensorView math will reflect the
//! series of split/merge/computeAts that have taken place, however these
//! nodes will not be displayed in what is printed. IrTransformPrinter does not
//! print any mathematical functions and only lists the series of
//! split/merge calls that were made. Both of these printing methods are
//! quite verbose on purpose as to show accurately what is represented in the IR
//! of a fusion.
//
//! \sa IrTransformPrinter
//!
class TORCH_CUDA_CU_API IrMathPrinter : public IrPrinter {
 public:
  IrMathPrinter(std::ostream& os) : IrPrinter(os) {}

  void handle(const Split* const) override {}
  void handle(const Merge* const) override {}

  void handle(Fusion* f) override {
    IrPrinter::handle(f);
  }
};

//! Prints transformation (schedule) Fusion IR nodes
//!
//! \sa IrMathPrinter
//!
class TORCH_CUDA_CU_API IrTransformPrinter : public IrPrinter {
 public:
  IrTransformPrinter(std::ostream& os) : IrPrinter(os) {}

  void handle(const UnaryOp* const uop) override {
    if (printInline()) {
      IrPrinter::handle(uop);
    }
  }

  void handle(const BinaryOp* const bop) override {
    if (printInline()) {
      IrPrinter::handle(bop);
    }
  }

  void handle(Fusion* f) override {
    IrPrinter::handle(f);
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
