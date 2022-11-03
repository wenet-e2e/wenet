#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Eliminates common inputs among `aten::cat` ops.
TORCH_API void EliminateConcatCommonInputs(const std::shared_ptr<Graph>& graph);

// Expands `aten::cat` ops into `aten::copy` ops and eliminates redudancies
// in the buffers used for concatenation if possible.
TORCH_API void ExpandConcatAndEliminateRedundancy(
    const std::shared_ptr<Graph>& graph);

// Performs all optimizations on `aten::cat` ops. Currently, this includes
//  * EliminateConcatCommonInputs
//  * ExpandConcatAndEliminateRedundancy
TORCH_API void OptimizeConcat(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
