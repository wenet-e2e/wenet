#pragma once
#include <ATen/Config.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/core/ScalarType.h>

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = acc_type<scalar_t, true>;

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

namespace at {

template <typename T, bool is_cuda>
struct AccumulateType { };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct AccumulateType<half, true> { using type = float; };
#endif
template <> struct AccumulateType<BFloat16, true> {using type = float; };
template <> struct AccumulateType<Half, true> { using type = float; };
template <> struct AccumulateType<float, true> { using type = float; };
template <> struct AccumulateType<double, true> { using type = double; };
template <> struct AccumulateType<int8_t, true> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, true> { using type = int64_t; };
template <> struct AccumulateType<char, true> { using type = int64_t; };
template <> struct AccumulateType<int16_t, true> { using type = int64_t; };
template <> struct AccumulateType<int32_t, true> { using type = int64_t; };
template <> struct AccumulateType<int64_t, true> { using type = int64_t; };
template <> struct AccumulateType<bool, true> {using type = bool; };
template <> struct AccumulateType<Half, false> { using type = float; };
template <> struct AccumulateType<BFloat16, false> { using type = float; };
template <> struct AccumulateType<c10::complex<float>, false> { using type = c10::complex<double>; };
template <> struct AccumulateType<c10::complex<double>, false> { using type = c10::complex<double>; };
template <> struct AccumulateType<c10::complex<float>, true> { using type = c10::complex<float>; };
template <> struct AccumulateType<c10::complex<double>, true> { using type = c10::complex<double>; };
template <> struct AccumulateType<float, false> { using type = double; };
template <> struct AccumulateType<double, false> { using type = double; };
template <> struct AccumulateType<int8_t, false> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, false> { using type = int64_t; };
template <> struct AccumulateType<char, false> { using type = int64_t; };
template <> struct AccumulateType<int16_t, false> { using type = int64_t; };
template <> struct AccumulateType<int32_t, false> { using type = int64_t; };
template <> struct AccumulateType<int64_t, false> { using type = int64_t; };
template <> struct AccumulateType<bool, false> {using type = bool; };

template<typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

}  // namespace at
