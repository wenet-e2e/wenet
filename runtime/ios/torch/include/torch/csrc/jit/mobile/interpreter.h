#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <vector>

namespace torch {
namespace jit {
namespace mobile {
using Stack = std::vector<c10::IValue>;
using DebugHandle = int64_t;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct Code {
  // TODO: Combine instructions and debug handles vector
  // into std::vector<<std::pair<Instruction, DebugHandle>>
  std::vector<Instruction> instructions_;
  std::vector<DebugHandle> debug_handles_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<std::function<void(Stack&)>> operators_;
  std::vector<c10::IValue> constants_;
  std::vector<c10::TypePtr> types_;
  size_t register_size_; // Aggregated output size.
};

struct InterpreterState {
  TORCH_API explicit InterpreterState(std::shared_ptr<Code> code);
  TORCH_API bool run(Stack& stack);

 private:
  std::shared_ptr<Code> code_;
  c10::IValue& reg(size_t reg);
  std::vector<c10::IValue> registers_;
};

// Interpreter executes instruction in a loop one by one
// from a list of instructions. PC is a program counter pointer
// pointing to the current instruction being executed.
// This function returns the current PC.
// Note that this is set only when exception occurs.
// since this is a thread local variable and setting it for
// every instruction will add overhead of thread local variable access.
int64_t getInterpretersExceptionPC();
} // namespace mobile
} // namespace jit
} // namespace torch
