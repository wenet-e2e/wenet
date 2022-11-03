#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <ATen/NumericUtils.h>
#include <limits>
#include <cstdint>
#include <cassert>

namespace at {

// Using DistAccumType in accumulate types for distributions.
// Note: Ideally we'd be using ATen/AccumulateType.h but looks
// like the there is some inconsistency in how accumulate types
// are mapped currently, e.g. for the cpu side, float is mapped
// to double.
template <typename T>
struct DistAccumType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct DistAccumType<half> { using type = float; };
#endif
template <> struct DistAccumType<BFloat16> { using type = float; };
template <> struct DistAccumType<Half> { using type = float; };
template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

namespace transformation {

/**
 * A transformation function for `torch.Tensor.random_()`, when both `from` and `to` are specified.
 * `range` is `to - from`
 * `base` is `from`
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when `from=min_value(int64_t)` and to=None
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_full_range(V val) {
  return static_cast<T>(static_cast<int64_t>(val));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`.
 * In order to prevent compiler warnings reported in GitHub issue 46391, T can't be float or double
 * in this overloaded version
 */
template <typename T, typename V>
C10_HOST_DEVICE inline typename std::enable_if<!(std::is_floating_point<T>::value), T>::type uniform_int(V val) {
  if (std::is_same<T, bool>::value) {
    return static_cast<bool>(val & 1);
  } else if (std::is_same<T, int64_t>::value) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else if (std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
  } else if (std::is_integral<T>::value) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else {
    assert(false);
    return 0;
  }
}

/**
 * An overloaded transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`,
 * added to fix compiler warnings reported in GitHub issue 46391. T is either float or double in this version.
 */
template<typename T, typename V>
C10_HOST_DEVICE inline typename std::enable_if<std::is_floating_point<T>::value, T>::type uniform_int(V val) {
  return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
}

template <typename T, typename V>
C10_HOST_DEVICE inline dist_acctype<T> uniform_real(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  dist_acctype<T> x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

/**
 * Transforms normally distributed `val` with mean 0.0 and standard deviation 1.0 to
 * normally distributed with `mean` and standard deviation `std`.
 */
template <typename T>
C10_HOST_DEVICE inline T normal(T val, T mean, T std) {
  return val * std + mean;
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * Cauchy distribution with location parameter `median` and scale parameter `sigma`.
 */
template <typename T>
C10_HOST_DEVICE inline T cauchy(T val, T median, T sigma) {
  // https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function
  return median + sigma * at::tan(c10::pi<T> * (val - static_cast<T>(0.5)));
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * exponentialy distributed with `lambda` parameter of the distribution.
 */
template <typename T>
C10_HOST_DEVICE __ubsan_ignore_float_divide_by_zero__ inline T exponential(T val, T lambda) {
  // https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
  // Different implementations for CUDA and CPU to preserve original logic
  // TODO: must be investigated and unified!!!
  // https://github.com/pytorch/pytorch/issues/38662
#if defined(__CUDACC__) || defined(__HIPCC__)
      // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
      // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
      // we need log to be not 0, and not underflow when converted to half
      // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1 args
  auto log = val >= static_cast<T>(1.) - std::numeric_limits<T>::epsilon() / 2
      ? -std::numeric_limits<T>::epsilon() / 2
      : at::log(val);
  return static_cast<T>(-1.0) / lambda * log;
#else
  return static_cast<T>(-1.0) / lambda * at::log(static_cast<T>(1.0) - val);
#endif
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * geometricaly distributed with success probability `p`.
 */
template <typename T>
C10_HOST_DEVICE inline T geometric(T val, T p) {
  // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
  return static_cast<T>(::ceil(at::log(val) / at::log(static_cast<T>(1.0) - p)));
}

/**
 * Transforms normally distributed `val` to log-normally distributed.
 */
template <typename T>
C10_HOST_DEVICE inline T log_normal(T val) {
  // https://en.wikipedia.org/wiki/Log-normal_distribution#Mode,_median,_quantiles
  return at::exp(val);
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * bernoulli distributed with success probability `p`.
 */
template <typename T>
C10_HOST_DEVICE inline T bernoulli(T val, T p) {
  return val < p;
}

}} // namespace at::transformation
