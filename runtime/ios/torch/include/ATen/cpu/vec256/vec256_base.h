#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
//
// Note [Do not compile initializers with AVX]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// If you define a static initializer in this file, the initialization will use
// AVX instructions because these object files are compiled with AVX enabled.
// We need to avoid non-trivial global data in these architecture specific files
// because there's no way to guard the global initializers with CPU capability
// detection.
//
// See https://github.com/pytorch/pytorch/issues/37577 for an instance
// of this bug in the past.

#include <cstring>
#include <functional>
#include <cmath>
#include <type_traits>
#include <bitset>

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/native/Math.h>
#include <ATen/NumericUtils.h>
#include <c10/util/C++17.h>
#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/copysign.h>
#include <c10/util/math_compat.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>

#if defined(__GNUC__)
#define __at_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align32__ __declspec(align(32))
#else
#define __at_align32__
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {
// at::Half and at::BFloat16 should be treated as floating point
template <typename T>
struct is_floating_point:
    std::integral_constant<bool,
      std::is_floating_point<T>::value ||
      std::is_same<T, at::Half>::value ||
      std::is_same<T, at::BFloat16>::value> {
};

template<size_t n> struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

// NOTE: If you specialize on a type, you must define all operations!

// emulates vectorized types
template <class T>
struct Vec256 {
private:
  __at_align32__ T values[32 / sizeof(T)];
public:
  using value_type = T;
  using size_type = int;
  // Note [constexpr static function to avoid odr-usage compiler bug]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Why, you might ask, is size defined to be a static constexpr function,
  // rather than a more ordinary 'static constexpr int size;' variable?
  // The problem lies within ODR rules for static constexpr members versus
  // static constexpr functions.  First, recall that this class (along with all
  // of its derivations) live in an anonymous namespace: they are intended to be
  // *completely* inlined at their use-sites, because we need to compile it
  // multiple times for different instruction sets.
  //
  // Because of this constraint, we CANNOT provide a single definition for
  // any static members in this class; since we want to compile the class
  // multiple times, there wouldn't actually be any good place to put the
  // definition.  Now here is the problem: if we ODR-use a static constexpr
  // member, we are *obligated* to provide a definition.  Without the
  // definition, you get a compile error like:
  //
  //    relocation R_X86_64_PC32 against undefined symbol
  //    `_ZN2at6vec25612_GLOBAL__N_16Vec256IdE4sizeE' can not be used when making
  //    a shared object; recompile with -fPIC
  //
  // If this were C++17, we could replace a static constexpr variable with
  // an inline variable which doesn't require one definition. But we are not
  // C++17.  So the next best thing is to replace the member with a static
  // constexpr (and therefore inline) function, which does not require ODR
  // either.
  //
  // Also, technically according to the C++ standard, we don't have to define
  // a constexpr variable if we never odr-use it.  But it seems that some
  // versions GCC/Clang have buggy determinations on whether or not an
  // identifier is odr-used or not, and in any case it's hard to tell if
  // a variable is odr-used or not.  So best to just cut the problem at the root.
  static constexpr size_type size() {
    return 32 / sizeof(T);
  }
  Vec256() : values{0} {}
  Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vec256(Args... vals) : values{vals...}{
  }
  // This also implies const T& operator[](int idx) const
  inline operator const T*() const {
    return values;
  }
  // This also implies T& operator[](int idx)
  inline operator T*() {
    return values;
  }
  template <int64_t mask_>
  static Vec256<T> blend(const Vec256<T>& a, const Vec256<T>& b) {
    int64_t mask = mask_;
    Vec256 vec;
    for (int64_t i = 0; i < size(); i++) {
      if (mask & 0x01) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vec;
  }
  static Vec256<T> blendv(const Vec256<T>& a, const Vec256<T>& b,
                          const Vec256<T>& mask) {
    Vec256 vec;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
    for (int64_t i = 0; i < size(); i++) {
      if (buffer[i] & 0x01)
       {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  template<typename step_t>  // step sometimes requires a higher precision type (e.g., T=int, step_t=double)
  static Vec256<T> arange(T base = static_cast<T>(0), step_t step = static_cast<step_t>(1)) {
    Vec256 vec;
    for (int64_t i = 0; i < size(); i++) {
      vec.values[i] = base + i * step;
    }
    return vec;
  }
  static Vec256<T> set(const Vec256<T>& a, const Vec256<T>& b, int64_t count = size()) {
    Vec256 vec;
    for (int64_t i = 0; i < size(); i++) {
      if (i < count) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<T> loadu(const void* ptr) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, 32);
    return vec;
  }
  static Vec256<T> loadu(const void* ptr, int64_t count) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(T));
    return vec;
  }
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int mask = 0;
    for (int i = 0; i < size(); ++ i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vec256<T> isnan() const {
    Vec256<T> vec;
    for (int64_t i = 0; i != size(); i++) {
      if (_isnan(values[i])) {
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(T));
      }
    }
    return vec;
  }
  Vec256<T> map(T (*f)(T)) const {
    Vec256<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<T> map(T (*f)(const T &)) const {
    Vec256<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  template <typename other_t_abs = T,
            typename std::enable_if<!is_floating_point<other_t_abs>::value && !c10::is_complex<other_t_abs>::value, int>::type = 0>
  Vec256<T> abs() const {
    // other_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_abs, T>::value, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  template <typename float_t_abs = T,
            typename std::enable_if<is_floating_point<float_t_abs>::value, int>::type = 0>
  Vec256<T> abs() const {
    // float_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<float_t_abs, T>::value, "float_t_abs must be T");
    // Specifically deal with floating-point because the generic code above won't handle -0.0 (which should result in
    // 0.0) properly.
    return map([](T x) -> T { return std::abs(x); });
  }
  template <typename complex_t_abs = T,
            typename std::enable_if<c10::is_complex<complex_t_abs>::value, int>::type = 0>
  Vec256<T> abs() const {
    // complex_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_abs, T>::value, "complex_t_abs must be T");
    // Specifically map() does not perform the type conversion needed by abs.
    return map([](T x) { return static_cast<T>(std::abs(x)); });
  }

  template <typename other_t_sgn = T,
            typename std::enable_if<c10::is_complex<other_t_sgn>::value, int>::type = 0>
  Vec256<T> sgn() const {
    return map(at::native::sgn_impl);
  }

  template <typename other_t_angle = T,
            typename std::enable_if<!c10::is_complex<other_t_angle>::value, int>::type = 0>
  Vec256<T> angle() const {
    // other_t_angle is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_angle, T>::value, "other_t_angle must be T");
    return map(at::native::angle_impl<T>);  // compiler is unable to resolve the overload without <T>
  }
  template <typename complex_t_angle = T,
            typename std::enable_if<c10::is_complex<complex_t_angle>::value, int>::type = 0>
  Vec256<T> angle() const {
    // complex_t_angle is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_angle, T>::value, "complex_t_angle must be T");
    return map([](T x) { return static_cast<T>(std::arg(x)); });
  }
  template <typename other_t_real = T,
            typename std::enable_if<!c10::is_complex<other_t_real>::value, int>::type = 0>
  Vec256<T> real() const {
    // other_t_real is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_real, T>::value, "other_t_real must be T");
    return *this;
  }
  template <typename complex_t_real = T,
            typename std::enable_if<c10::is_complex<complex_t_real>::value, int>::type = 0>
  Vec256<T> real() const {
    // complex_t_real is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_real, T>::value, "complex_t_real must be T");
    return map([](T x) { return static_cast<T>(x.real()); });
  }
  template <typename other_t_imag = T,
            typename std::enable_if<!c10::is_complex<other_t_imag>::value, int>::type = 0>
  Vec256<T> imag() const {
    // other_t_imag is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_imag, T>::value, "other_t_imag must be T");
    return Vec256(0);
  }
  template <typename complex_t_imag = T,
            typename std::enable_if<c10::is_complex<complex_t_imag>::value, int>::type = 0>
  Vec256<T> imag() const {
    // complex_t_imag is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_imag, T>::value, "complex_t_imag must be T");
    return map([](T x) { return static_cast<T>(x.imag()); });
  }
  template <typename other_t_conj = T,
            typename std::enable_if<!c10::is_complex<other_t_conj>::value, int>::type = 0>
  Vec256<T> conj() const {
    // other_t_conj is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_conj, T>::value, "other_t_conj must be T");
    return *this;
  }
  template <typename complex_t_conj = T,
            typename std::enable_if<c10::is_complex<complex_t_conj>::value, int>::type = 0>
  Vec256<T> conj() const {
    // complex_t_conj is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_conj, T>::value, "complex_t_conj must be T");
    return map([](T x) { return static_cast<T>(std::conj(x)); });
  }
  Vec256<T> acos() const {
    return map(std::acos);
  }
  Vec256<T> asin() const {
    return map(std::asin);
  }
  Vec256<T> atan() const {
    return map(std::atan);
  }
  Vec256<T> atan2(const Vec256<T> &exp) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::atan2(values[i], exp[i]);
    }
    return ret;
  }
  template <
    typename U = T,
    typename std::enable_if_t<is_floating_point<U>::value, int> = 0>
  Vec256<T> copysign(const Vec256<T> &sign) const {
    Vec256<T> ret;
    for (size_type i = 0; i < size(); i++) {
      ret[i] = c10::copysign(values[i], sign[i]);
    }
    return ret;
  }
  Vec256<T> erf() const {
    return map(std::erf);
  }
  Vec256<T> erfc() const {
    return map(std::erfc);
  }
  Vec256<T> erfinv() const {
    return map(calc_erfinv);
  }
  Vec256<T> exp() const {
    return map(std::exp);
  }
  Vec256<T> expm1() const {
    return map(std::expm1);
  }
  Vec256<T> frac() const {
    return *this - this->trunc();
  }
  template <
    typename U = T,
    typename std::enable_if_t<is_floating_point<U>::value, int> = 0>
  Vec256<T> fmod(const Vec256<T>& q) const {
    // U is for SFINAE purposes only. Make sure it is not changed.
    static_assert(std::is_same<U, T>::value, "U must be T");
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::fmod(values[i], q[i]);
    }
    return ret;
  }
  Vec256<T> log() const {
    return map(std::log);
  }
  Vec256<T> log10() const {
    return map(std::log10);
  }
  Vec256<T> log1p() const {
    return map(std::log1p);
  }
  template <typename other_t_log2 = T,
            typename std::enable_if<!c10::is_complex<other_t_log2>::value, int>::type = 0>
  Vec256<T> log2() const {
    // other_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_log2, T>::value, "other_t_log2 must be T");
    return map(std::log2);
  }
  template <typename complex_t_log2 = T,
            typename std::enable_if<c10::is_complex<complex_t_log2>::value, int>::type = 0>
  Vec256<T> log2() const {
    // complex_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<complex_t_log2, T>::value, "complex_t_log2 must be T");
    const T log_2 = T(std::log(2.0));
    return Vec256(map(std::log))/Vec256(log_2);
  }
  Vec256<T> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vec256<T> cos() const {
    return map(std::cos);
  }
  Vec256<T> cosh() const {
    return map(std::cosh);
  }
  Vec256<T> floor() const {
    return map(at::native::floor_impl);
  }
  Vec256<T> hypot(const Vec256<T> &b) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::hypot(values[i], b[i]);
    }
    return ret;
  }
  Vec256<T> i0() const {
    return map(calc_i0);
  }
  Vec256<T> i0e() const {
    return map(calc_i0e);
  }
  Vec256<T> igamma(const Vec256<T> &x) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = calc_igamma(values[i], x[i]);
    }
    return ret;
  }
  Vec256<T> igammac(const Vec256<T> &x) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = calc_igammac(values[i], x[i]);
    }
    return ret;
  }
  Vec256<T> neg() const {
    // NB: the trailing return type is needed because we need to coerce the
    // return value back to T in the case of unary operator- incuring a
    // promotion
    return map([](T x) -> T { return -x; });
  }
  Vec256<T> nextafter(const Vec256<T> &b) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::nextafter(values[i], b[i]);
    }
    return ret;
  }
  Vec256<T> round() const {
    // We do not use std::round because we would like to round midway numbers to the nearest even integer.
    return map(at::native::round_impl);
  }
  Vec256<T> sin() const {
    return map(std::sin);
  }
  Vec256<T> sinh() const {
    return map(std::sinh);
  }
  Vec256<T> tan() const {
    return map(std::tan);
  }
  Vec256<T> tanh() const {
    return map(std::tanh);
  }
  Vec256<T> trunc() const {
    return map(at::native::trunc_impl);
  }
  Vec256<T> lgamma() const {
    return map(std::lgamma);
  }
  Vec256<T> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vec256<T> rsqrt() const {
    return map([](T x) { return (T)1 / std::sqrt(x); });
  }
  Vec256<T> pow(const Vec256<T> &exp) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
private:
  template <typename Op>
  inline Vec256<T> binary_pred(const Vec256<T>& other, Op op) const {
    // All bits are set to 1 if the pred is true, otherwise 0.
    Vec256<T> vec;
    for (int64_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(T));
      }
    }
    return vec;
  }

public:
  Vec256<T> operator==(const Vec256<T>& other) const { return binary_pred(other, std::equal_to<T>()); }
  Vec256<T> operator!=(const Vec256<T>& other) const { return binary_pred(other, std::not_equal_to<T>()); }
  Vec256<T> operator>=(const Vec256<T>& other) const { return binary_pred(other, std::greater_equal<T>()); }
  Vec256<T> operator<=(const Vec256<T>& other) const { return binary_pred(other, std::less_equal<T>()); }
  Vec256<T> operator>(const Vec256<T>& other) const { return binary_pred(other, std::greater<T>()); }
  Vec256<T> operator<(const Vec256<T>& other) const { return binary_pred(other, std::less<T>()); }

private:
  template <typename Op>
  inline Vec256<T> binary_pred_bool(const Vec256<T>& other, Op op) const {
    // 1 if the pred is true, otherwise 0.
    Vec256<T> vec;
    for (int i = 0; i != size(); ++ i) {
      vec[i] = bool(op(values[i], other.values[i]));
    }
    return vec;
  }

public:
  Vec256<T> eq(const Vec256<T>& other) const { return binary_pred_bool(other, std::equal_to<T>()); }
  Vec256<T> ne(const Vec256<T>& other) const { return binary_pred_bool(other, std::not_equal_to<T>()); }
  Vec256<T> gt(const Vec256<T>& other) const { return binary_pred_bool(other, std::greater<T>()); }
  Vec256<T> ge(const Vec256<T>& other) const { return binary_pred_bool(other, std::greater_equal<T>()); }
  Vec256<T> lt(const Vec256<T>& other) const { return binary_pred_bool(other, std::less<T>()); }
  Vec256<T> le(const Vec256<T>& other) const { return binary_pred_bool(other, std::less_equal<T>()); }
};

template <class T> Vec256<T> inline operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator-(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator/(const Vec256<T> &a, const Vec256<T> &b) __ubsan_ignore_float_divide_by_zero__ {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator||(
    const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] || b[i];
  }
  return c;
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <class T,
          typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline maximum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if<c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline maximum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (std::abs(a[i]) > std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <class T,
          typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline minimum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if<c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline minimum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline clamp(const Vec256<T> &a, const Vec256<T> &min_vec, const Vec256<T> &max_vec) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c;
}

template <class T,
          typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline clamp_max(const Vec256<T> &a, const Vec256<T> &max_vec) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c;
}

template <class T,
          typename std::enable_if<!c10::is_complex<T>::value, int>::type = 0>
Vec256<T> inline clamp_min(const Vec256<T> &a, const Vec256<T> &min_vec) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

struct Vec256i;

#ifdef CPU_CAPABILITY_AVX2

template <class T, typename Op>
static inline Vec256<T> bitwise_binary_op(const Vec256<T> &a, const Vec256<T> &b, Op op) {
  __m256i buffer;
  __m256i a_buffer = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const T*)a));
  __m256i b_buffer = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((const T*)b));
  buffer = op(a_buffer, b_buffer);
  __at_align32__ T results[Vec256<T>::size()];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), buffer);
  return Vec256<T>::loadu(results);
}

template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator&(const Vec256<T>& a, const Vec256<T>& b) {
  // We enclose _mm256_and_si256 with lambda because it is always_inline
  return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_and_si256(a, b); });
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator|(const Vec256<T>& a, const Vec256<T>& b) {
  // We enclose _mm256_or_si256 with lambda because it is always_inline
  return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_or_si256(a, b); });
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator^(const Vec256<T>& a, const Vec256<T>& b) {
  // We enclose _mm256_xor_si256 with lambda because it is always_inline
  return bitwise_binary_op(a, b, [](__m256i a, __m256i b) { return _mm256_xor_si256(a, b); });
}

#else

template<class T, typename Op>
static inline Vec256<T> bitwise_binary_op(const Vec256<T> &a, const Vec256<T> &b, Op op) {
  static constexpr uint32_t element_no = 32 / sizeof(intmax_t);
  __at_align32__ intmax_t buffer[element_no];
  const intmax_t *a_ptr = reinterpret_cast<const intmax_t*>((const T*) a);
  const intmax_t *b_ptr = reinterpret_cast<const intmax_t*>((const T*) b);
  for (uint32_t i = 0U; i < element_no; ++ i) {
    buffer[i] = op(a_ptr[i], b_ptr[i]);
  }
  return Vec256<T>::loadu(buffer);
}

template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator&(const Vec256<T>& a, const Vec256<T>& b) {
  return bitwise_binary_op(a, b, std::bit_and<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator|(const Vec256<T>& a, const Vec256<T>& b) {
  return bitwise_binary_op(a, b, std::bit_or<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator^(const Vec256<T>& a, const Vec256<T>& b) {
  return bitwise_binary_op(a, b, std::bit_xor<intmax_t>());
}

#endif

template<class T, typename std::enable_if_t<!std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator~(const Vec256<T>& a) {
  Vec256<T> ones;  // All bits are 1
  memset((T*) ones, 0xFF, 32);
  return a ^ ones;
}


template <typename T>
inline Vec256<T>& operator += (Vec256<T>& a, const Vec256<T>& b) {
  a = a + b;
  return a;
}
template <typename T>
inline Vec256<T>& operator -= (Vec256<T>& a, const Vec256<T>& b) {
  a = a - b;
  return a;
}
template <typename T>
inline Vec256<T>& operator /= (Vec256<T>& a, const Vec256<T>& b) {
  a = a / b;
  return a;
}
template <typename T>
inline Vec256<T>& operator %= (Vec256<T>& a, const Vec256<T>& b) {
  a = a % b;
  return a;
}
template <typename T>
inline Vec256<T>& operator *= (Vec256<T>& a, const Vec256<T>& b) {
  a = a * b;
  return a;
}

template <typename T>
inline Vec256<T> fmadd(const Vec256<T>& a, const Vec256<T>& b, const Vec256<T>& c) {
  return a * b + c;
}

template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<T>>
inline gather(T const* base_addr, const Vec256<int_same_size_t<T>>& vindex) {
  static constexpr int size = Vec256<T>::size();
  int_same_size_t<T> index_arr[size];
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (int64_t i = 0; i < size; i++) {
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
  }
  return Vec256<T>::loadu(static_cast<void*>(buffer));
}

template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<T>>
inline mask_gather(const Vec256<T>& src, T const* base_addr,
                   const Vec256<int_same_size_t<T>>& vindex, Vec256<T>& mask) {
  static constexpr int size = Vec256<T>::size();
  T src_arr[size];
  int_same_size_t<T> mask_arr[size];  // use int type so we can logical and
  int_same_size_t<T> index_arr[size];
  src.store(static_cast<void*>(src_arr));
  mask.store(static_cast<void*>(mask_arr));
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (int64_t i = 0; i < size; i++) {
    if (mask_arr[i] & 0x01) {  // check highest bit
      buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
    } else {
      buffer[i] = src_arr[i];
    }
  }
  mask = Vec256<T>();  // "zero out" mask
  return Vec256<T>::loadu(static_cast<void*>(buffer));
}

// Cast a given vector to another type without changing the bits representation.
// So a Vec<double> of 256 bits containing all ones can be cast to a
// Vec<int64_t> of 256 bits containing all ones (i.e., four negative 1s).
namespace {
  // There is a struct here because we don't have static_if and I can't
  // partially specialize a templated function.
  template<typename dst_t, typename src_t>
  struct CastImpl {
    static inline Vec256<dst_t> apply(const Vec256<src_t>& src) {
      src_t src_arr[Vec256<src_t>::size()];
      src.store(static_cast<void*>(src_arr));
      return Vec256<dst_t>::loadu(static_cast<const void*>(src_arr));
    }
  };

  template<typename scalar_t>
  struct CastImpl<scalar_t, scalar_t> {
    static inline Vec256<scalar_t> apply(const Vec256<scalar_t>& src) {
      return src;
    }
  };
}
template<typename dst_t, typename src_t>
inline Vec256<dst_t> cast(const Vec256<src_t>& src) {
  return CastImpl<dst_t, src_t>::apply(src);
}

template <typename T>
inline Vec256<int_same_size_t<T>> convert_to_int_of_same_size(const Vec256<T>& src) {
  static constexpr int size = Vec256<T>::size();
  T src_arr[size];
  src.store(static_cast<void*>(src_arr));
  int_same_size_t<T> buffer[size];
  for (int64_t i = 0; i < size; i++) {
    buffer[i] = static_cast<int_same_size_t<T>>(src_arr[i]);
  }
  return Vec256<int_same_size_t<T>>::loadu(static_cast<void*>(buffer));
}

// E.g., inputs: a           Vec256<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//               b           Vec256<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
//       returns:            Vec256<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//                           Vec256<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
template <typename T>
inline std::enable_if_t<Vec256<T>::size() % 2 == 0, std::pair<Vec256<T>, Vec256<T>>>
deinterleave2(const Vec256<T>& a, const Vec256<T>& b) {
  static constexpr int size = Vec256<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (int64_t i = 0; i < half_size; i++) {
    buffer1[i] = a_arr[i * 2];
    buffer1[half_size + i] = b_arr[i * 2];
    buffer2[i] = a_arr[i * 2 + 1];
    buffer2[half_size + i] = b_arr[i * 2 + 1];
  }
  return std::make_pair(Vec256<T>::loadu(static_cast<void*>(buffer1)),
                        Vec256<T>::loadu(static_cast<void*>(buffer2)));
}

// inverse operation of deinterleave2
// E.g., inputs: a           Vec256<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//               b           Vec256<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
//       returns:            Vec256<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//                           Vec256<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
template <typename T>
inline std::enable_if_t<Vec256<T>::size() % 2 == 0, std::pair<Vec256<T>, Vec256<T>>>
interleave2(const Vec256<T>& a, const Vec256<T>& b) {
  static constexpr int size = Vec256<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (int64_t i = 0; i < half_size; i++) {
    buffer1[i * 2] = a_arr[i];
    buffer1[i * 2 + 1] = b_arr[i];
    buffer2[i * 2] = a_arr[half_size + i];
    buffer2[i * 2 + 1] = b_arr[half_size + i];
  }
  return std::make_pair(Vec256<T>::loadu(static_cast<void*>(buffer1)),
                        Vec256<T>::loadu(static_cast<void*>(buffer2)));
}

template <typename src_T, typename dst_T>
inline void convert(const src_T *src, dst_T *dst, int64_t n) {
#ifndef _MSC_VER
# pragma unroll
#endif
  for (int64_t i = 0; i < n; i++) {
    *dst = c10::static_cast_with_inter_type<dst_T, src_T>::apply(*src);
    src++;
    dst++;
  }
}

}}}
