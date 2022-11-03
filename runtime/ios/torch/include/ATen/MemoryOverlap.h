#pragma once

#include <ATen/ATen.h>

namespace at {

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD };

enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

TORCH_API MemOverlap has_internal_overlap(const Tensor& t);
TORCH_API MemOverlap has_internal_overlap(TensorImpl* t);

TORCH_API void assert_no_internal_overlap(const Tensor& t);
TORCH_API void assert_no_internal_overlap(TensorImpl* t);

TORCH_API MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b);
TORCH_API MemOverlapStatus get_overlap_status(TensorImpl* a, TensorImpl* b);

TORCH_API void assert_no_partial_overlap(const Tensor& a, const Tensor& b);
void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b);

TORCH_API void assert_no_overlap(const Tensor& a, const Tensor& b);
TORCH_API void assert_no_overlap(TensorImpl* a, TensorImpl* b);

}
