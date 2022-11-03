#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * Mutators are the mechanism used to modify IR nodes. Since most nodes are
 * immutable or at least partially immutable changeing them can require creating
 * a new node. Base mutator at the moment is a dumb sample mutator that takes
 * any float of value 1.0 and converts it to 0.0; It is currently used as a
 * dummy example, however, we should make it a simple instantiation of all the
 * mutate functions on all node types so that people can inhereit it, and only
 * specialize those nodes which they want to have a particular transformation.
 */

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
