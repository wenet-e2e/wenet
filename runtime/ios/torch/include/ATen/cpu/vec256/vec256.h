#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/intrinsics.h>

#include <ATen/cpu/vec256/vec256_base.h>
#if !defined(__VSX__)  || !defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec256/vec256_float.h>
#include <ATen/cpu/vec256/vec256_float_neon.h>
#include <ATen/cpu/vec256/vec256_bfloat16.h>
#include <ATen/cpu/vec256/vec256_double.h>
#include <ATen/cpu/vec256/vec256_int.h>
#include <ATen/cpu/vec256/vec256_qint.h>
#include <ATen/cpu/vec256/vec256_complex_float.h>
#include <ATen/cpu/vec256/vec256_complex_double.h>
#else
#include <ATen/cpu/vec256/vsx/vec256_common_vsx.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace at {
namespace vec256 {

// Note [Acceptable use of anonymous namespace in header]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Yes you saw right, this is an anonymous namespace in a header.  This header,
// and all of its subheaders, REQUIRE their code to be entirely inlined into
// the compilation unit that uses them.  It's important that these functions have
// internal linkage so that kernels for different architectures don't get
// combined during linking. It's sufficient to label functions "static", but
// class methods must be an unnamed namespace to have internal linkage (since
// static means something different in the context of classes).
namespace {

 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
     stream << val.val_;
     return stream;
 }
 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
     stream << static_cast<int>(val.val_);
     return stream;
 }
 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
     stream << static_cast<unsigned int>(val.val_);
     return stream;
 }

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vec256<T>& vec) {
  T buf[Vec256<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != Vec256<T>::size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}


#if (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vec256<float> cast<float, double>(const Vec256<double>& src) {
  return _mm256_castpd_ps(src);
}

template<>
inline Vec256<double> cast<double, float>(const Vec256<float>& src) {
  return _mm256_castps_pd(src);
}

#if defined(CPU_CAPABILITY_AVX2)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define DEFINE_FLOAT_INT_CAST(int_t, float_t, float_ch)            \
template<>                                                         \
inline  Vec256<int_t> cast<int_t, float_t>(const Vec256<float_t>& src) {   \
  return _mm256_castp ## float_ch ## _si256(src);                  \
}                                                                  \
template<>                                                         \
inline Vec256<float_t> cast<float_t, int_t>(const Vec256<int_t>& src) {   \
  return _mm256_castsi256_p ## float_ch (src);                     \
}

DEFINE_FLOAT_INT_CAST(int64_t, double, d)
DEFINE_FLOAT_INT_CAST(int32_t, double, d)
DEFINE_FLOAT_INT_CAST(int16_t, double, d)
DEFINE_FLOAT_INT_CAST(int64_t, float, s)
DEFINE_FLOAT_INT_CAST(int32_t, float, s)
DEFINE_FLOAT_INT_CAST(int16_t, float, s)

#undef DEFINE_FLOAT_INT_CAST

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<double>>
inline gather(const double* base_addr, const Vec256<int64_t>& vindex) {
  return _mm256_i64gather_pd(base_addr, vindex, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<float>>
inline gather(const float* base_addr, const Vec256<int32_t>& vindex) {
  return _mm256_i32gather_ps(base_addr, vindex, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<double>>
inline mask_gather(const Vec256<double>& src, const double* base_addr,
                   const Vec256<int64_t>& vindex, const Vec256<double>& mask) {
  return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vec256<float>>
inline mask_gather(const Vec256<float>& src, const float* base_addr,
                   const Vec256<int32_t>& vindex, const Vec256<float>& mask) {
  return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Only works for inputs in the range: [-2^51, 2^51]
// From: https://stackoverflow.com/a/41148578
template<>
Vec256<int64_t>
inline convert_to_int_of_same_size<double>(const Vec256<double> &src) {
  auto x = _mm256_add_pd(src, _mm256_set1_pd(0x0018000000000000));
  return _mm256_sub_epi64(
      _mm256_castpd_si256(x),
      _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
  );
}

template<>
Vec256<int32_t>
inline convert_to_int_of_same_size<float>(const Vec256<float> &src) {
  return _mm256_cvttps_epi32(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vec256<double>, Vec256<double>>
inline interleave2<double>(const Vec256<double>& a, const Vec256<double>& b) {
  // inputs:
  //   a = {a0, a1, a3, a3}
  //   b = {b0, b1, b2, b3}

  // swap lanes:
  //   a_swapped = {a0, a1, b0, b1}
  //   b_swapped = {a2, a3, b2, b3}
  auto a_swapped = _mm256_permute2f128_pd(a, b, 0b0100000);  // 0, 2.   4 bits apart
  auto b_swapped = _mm256_permute2f128_pd(a, b, 0b0110001);  // 1, 3.   4 bits apart

  // group cols crossing lanes:
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(_mm256_permute4x64_pd(a_swapped, 0b11011000),  // 0, 2, 1, 3
                        _mm256_permute4x64_pd(b_swapped, 0b11011000)); // 0, 2, 1, 3
}

template <>
std::pair<Vec256<float>, Vec256<float>>
inline interleave2<float>(const Vec256<float>& a, const Vec256<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}

  // swap lanes:
  //   a_swapped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_swapped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: can we support caching this?
  auto a_swapped = _mm256_permute2f128_ps(a, b, 0b0100000);  // 0, 2.   4 bits apart
  auto b_swapped = _mm256_permute2f128_ps(a, b, 0b0110001);  // 1, 3.   4 bits apart

  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  return std::make_pair(_mm256_permutevar8x32_ps(a_swapped, group_ctrl),
                        _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vec256<double>, Vec256<double>>
inline deinterleave2<double>(const Vec256<double>& a, const Vec256<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}

  // group cols crossing lanes:
  //   a_grouped = {a0, a1, b0, b1}
  //   b_grouped = {a2, a3, b2, b3}
  auto a_grouped = _mm256_permute4x64_pd(a, 0b11011000);  // 0, 2, 1, 3
  auto b_grouped = _mm256_permute4x64_pd(b, 0b11011000);  // 0, 2, 1, 3

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(_mm256_permute2f128_pd(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                        _mm256_permute2f128_pd(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

template <>
std::pair<Vec256<float>, Vec256<float>>
inline deinterleave2<float>(const Vec256<float>& a, const Vec256<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}

  // group cols crossing lanes:
  //   a_grouped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_grouped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: can we support caching this?
  const __m256i group_ctrl = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  auto a_grouped = _mm256_permutevar8x32_ps(a, group_ctrl);
  auto b_grouped = _mm256_permutevar8x32_ps(b, group_ctrl);

  // swap lanes:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  return std::make_pair(_mm256_permute2f128_ps(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                        _mm256_permute2f128_ps(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

#endif  // defined(CPU_CAPABILITY_AVX2)

#endif // (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

}}}
