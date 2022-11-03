#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/native/quantized/affine_quantizer_base.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>

#include <array>

// This file defines Vec256<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vec256<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vec256<qint8> -> 4x Vec256<float>
//  Vec256<quint8> -> 4x Vec256<float>
//  Vec256<qint32> -> 1x Vec256<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vec256<T>::float_num_vecs
// iterations.

namespace at {
namespace vec256 {
namespace {

#if (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

struct Vec256qi {
 protected:
  __m256i vals __attribute__((aligned(64)));

 public:
  Vec256qi() {}
  Vec256qi(__m256i v) : vals(v) {}
  operator __m256i() const {
    return vals;
  }
};

#if defined(CPU_CAPABILITY_AVX2)
template <typename T>
__m256i pack_saturate_and_clamp(
    __m256i first,
    __m256i second,
    T min_val,
    T max_val);

template <>
__m256i pack_saturate_and_clamp<int32_t>(
    __m256i first,
    __m256i second,
    int32_t min_val,
    int32_t max_val) {
  // This function is for linkage only, will not be used
  AT_ERROR("pack_saturate_and_clamp<int32_t> is not supported");
}

template <>
__m256i pack_saturate_and_clamp<int8_t>(
    __m256i first,
    __m256i second,
    int8_t min_val,
    int8_t max_val) {
  __m256i packed_and_sat = _mm256_packs_epi16(first, second);
  return _mm256_max_epi8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epi8(packed_and_sat, _mm256_set1_epi8(max_val)));
}

template <>
__m256i pack_saturate_and_clamp<uint8_t>(
    __m256i first,
    __m256i second,
    uint8_t min_val,
    uint8_t max_val) {
  __m256i packed_and_sat = _mm256_packus_epi16(first, second);
  return _mm256_max_epu8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epu8(packed_and_sat, _mm256_set1_epi8(max_val)));
}
#endif

template <typename T>
inline void __attribute__((always_inline)) QuantizeAvx2(
    const float* src,
    typename T::underlying* dst,
    int len,
    float inverse_scale,
    int64_t zero_point) {
#if defined(CPU_CAPABILITY_AVX2)
  constexpr int VLEN = 8;
  constexpr auto min_val = std::numeric_limits<typename T::underlying>::min();
  constexpr auto max_val = std::numeric_limits<typename T::underlying>::max();
  const __m256i min_v = _mm256_set1_epi32(min_val);
  const __m256i max_v = _mm256_set1_epi32(max_val);
  // This is the largest int32 value < int32_max exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  // clang-format off
  static const __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00);
  // clang-format on
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i permute_mask_l8_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  for (; i < len_aligned; i += 4 * VLEN) {
    // x
    __m256 x_vals = _mm256_load_ps(src + i);
    __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm256_cvtps_epi32 converts them to -ve. Clip at int32_float_max_val to
    // Clip at int32_float_max_val to avoid this.
    x_transformed_v =
        _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // y
    __m256 y_vals = _mm256_load_ps(src + i + VLEN);
    __m256 y_transformed_v = _mm256_mul_ps(y_vals, inverse_scale_v);
    y_transformed_v =
        _mm256_min_ps(y_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // z
    __m256 z_vals = _mm256_load_ps(src + i + 2 * VLEN);
    __m256 z_transformed_v = _mm256_mul_ps(z_vals, inverse_scale_v);
    z_transformed_v =
        _mm256_min_ps(z_transformed_v, _mm256_set1_ps(int32_float_max_val));
    // w
    __m256 w_vals = _mm256_load_ps(src + i + 3 * VLEN);
    __m256 w_transformed_v = _mm256_mul_ps(w_vals, inverse_scale_v);
    w_transformed_v =
        _mm256_min_ps(w_transformed_v, _mm256_set1_ps(int32_float_max_val));

    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_transformed_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_transformed_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_transformed_v);

    // add zero point
    x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
    y_rounded_v = _mm256_add_epi32(y_rounded_v, _mm256_set1_epi32(zero_point));
    z_rounded_v = _mm256_add_epi32(z_rounded_v, _mm256_set1_epi32(zero_point));
    w_rounded_v = _mm256_add_epi32(w_rounded_v, _mm256_set1_epi32(zero_point));

    __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
    __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
    __m256i xyzw_clamped_v = pack_saturate_and_clamp<typename T::underlying>(
        xy_packed_v, zw_packed_v, min_val, max_val);

    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), xyzw_clamped_v);
  }

  // Additional 8-lane AVX2 version to take advantage when len is smaller
  // based on fbgemm::QuantizeAvx2 (https://github.com/pytorch/FBGEMM)
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m256 x_vals = _mm256_load_ps(src + i);
    __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
    x_transformed_v =
        _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
    __m256i x_clipped_v =
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, x_rounded_v));

    x_clipped_v = _mm256_shuffle_epi8(x_clipped_v, shuffle_mask_v);
    x_clipped_v = _mm256_permutevar8x32_epi32(x_clipped_v, permute_mask_l8_v);
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + i),
        _mm256_castsi256_si128(x_clipped_v));
  }

  for (; i < len; ++i) {
    float transformed = src[i] * inverse_scale;

    // Not exactly the same behavior as the vectorized code.
    // The vectorized code above always rounds to even in halfway cases
    // (https://software.intel.com/en-us/node/523819), but std::nearbyint
    // does the same only when the current rounding mode is FE_TONEAREST.
    // However, in practice, this should not be a problem because most cases
    // use the default rounding mode FE_TONEAREST.
    // Note that we cannot implement the same behavior as the vectorized code
    // using std::round because it does rounding away from zero in halfway
    // cases.
    transformed = zero_point + nearbyint(transformed);
    float clipped =
        std::min(std::max(transformed, float(min_val)), float(max_val));
    dst[i] = clipped;
  }
#else
  at::native::quantize_vec<T>(
      1.0f / inverse_scale, zero_point, src, reinterpret_cast<T*>(dst), len);
#endif
}

template<>
struct Vec256<c10::qint32> : public Vec256qi {
    using size_type = int;
    static constexpr size_type size() {
        return 8;
    }

    static constexpr int float_num_vecs() {
        return 1;
    }

    static constexpr int int_num_vecs() {
        return 1;
    }

    using float_vec_return_type = std::array<Vec256<float>, 1>;
    using int_vec_return_type = std::array<Vec256<c10::qint32>, 1>;
    using value_type = c10::qint32::underlying;

 public:
    using Vec256qi::Vec256qi;
    Vec256() {}

    Vec256(__m256i vals_) { vals = vals_;}

    // Broadcast constructor
    Vec256(const c10::qint32& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi32(uw);
    }

    void store(void* ptr, int count = size()) const {
      if (count != size()) {
        memcpy(ptr, &vals, count * sizeof(value_type));
      } else {
        _mm256_storeu_si256((__m256i*)ptr, vals);
      }
    }

    static Vec256<c10::qint32> loadu(const void* ptr) {
        return Vec256<c10::qint32>(ptr);
    }

    float_vec_return_type dequantize(
        Vec256<float> scale,
        Vec256<float> zero_point,
        Vec256<float> scale_zp_premul) const {
      __m256 float_vals = _mm256_cvtepi32_ps(vals);
#if defined(CPU_CAPABILITY_AVX2)
      return {vec256::fmadd(scale, Vec256<float>(float_vals), scale_zp_premul)};
#else
      return {scale * (Vec256<float>(float_vals) - zero_point)};
#endif
    }

    static Vec256<c10::qint32> quantize(
        const float_vec_return_type& rhs,
        float scale,
        int32_t zero_point,
        float inverse_scale) {
      Vec256<c10::qint32> retval;
      auto rhs_data = (__m256)rhs[0];
      at::native::quantize_vec<c10::qint32, /*precision=*/32>(
          scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 8);
      return retval;
    }

    Vec256<c10::qint32> maximum(Vec256<c10::qint32> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_max_epi32(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int32_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(int_vals.data()), vals);
      std::array<int32_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(b_vals.data()), b.vals);
      std::array<int32_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<int32_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::qint32> minimum(Vec256<c10::qint32> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epi32(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int32_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int32_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<int32_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int32_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::qint32> relu6(
        Vec256<c10::qint32> zero_point,
        Vec256<c10::qint32> q_six) {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epi32(
          _mm256_max_epi32(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int32_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int32_t, size()> zero_point_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      std::array<int32_t,size()> q_six_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      std::array<int32_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int32_t>(
            std::max<int32_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    int_vec_return_type widening_subtract(Vec256<c10::qint32> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return {_mm256_sub_epi32(vals, b)};
#else
      std::array<int32_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int32_t, size()> b_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<int32_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = int_vals[i] - b_vals[i];
      }
      return {_mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals))};
#endif
    }

    static Vec256<c10::qint32> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
#ifdef CPU_CAPABILITY_AVX2
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);

      __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier_v);
      __m256i rounded = _mm256_cvtps_epi32(scaled);
      return _mm256_add_epi32(rounded, zero_point_v);
#else
      std::array<int32_t,size()> inp_vals;
      inp[0].store(inp_vals.data());
      std::array<int32_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] =
            nearbyint(static_cast<float>(inp_vals[i]) * multiplier) +
            zero_point;
      }
      return loadu(result_vals.data());
#endif
    }

    void dump() const {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << ((int32_t*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    // Load from memory constructor
    Vec256(const void* ptr) {
      vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::qint32> inline maximum(const Vec256<c10::qint32>& a, const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vec256<c10::qint32> inline operator*(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
#ifdef CPU_CAPABILITY_AVX2
  return _mm256_mullo_epi32(a, b);
#else
  // Pray the compiler can autovectorize this
  std::array<int32_t, std::decay_t<decltype(a)>::size()> a_vals;
  std::array<int32_t, std::decay_t<decltype(b)>::size()> b_vals;
  a.store(a_vals.data());
  b.store(b_vals.data());
  std::array<int32_t, std::decay_t<decltype(a)>::size()> result_vals;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    result_vals[i] = a_vals[i] * b_vals[i];
  }
  return Vec256<c10::qint32>::loadu(result_vals.data());
#endif
}

template <>
Vec256<c10::qint32> inline operator+(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
#ifdef CPU_CAPABILITY_AVX2
  return _mm256_add_epi32(a, b);
#else
  // Pray the compiler can autovectorize this
  std::array<int32_t, std::decay_t<decltype(a)>::size()> a_vals;
  std::array<int32_t, std::decay_t<decltype(b)>::size()> b_vals;
  a.store(a_vals.data());
  b.store(b_vals.data());
  std::array<int32_t, std::decay_t<decltype(a)>::size()> result_vals;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    result_vals[i] = a_vals[i] + b_vals[i];
  }
  return Vec256<c10::qint32>::loadu(result_vals.data());
#endif
}

#ifdef CPU_CAPABILITY_AVX2
/*
 * Convert values from int32 back to int8/uint8
 */
template <typename T>
__m256i RequantizeAvx2(
    const std::array<Vec256<c10::qint32>, 4>& inp,
    __m256 multiplier,
    __m256i zp) {
  static_assert(
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
      "Only int8_t/uint8_t are supported");
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier);
  __m256 y_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[1]), multiplier);
  __m256 z_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[2]), multiplier);
  __m256 w_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[3]), multiplier);

  __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
  __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
  __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
  __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

  /* Add zero point */
  __m256i x_v = _mm256_add_epi32(x_rounded_v, zp);
  __m256i y_v = _mm256_add_epi32(y_rounded_v, zp);
  __m256i z_v = _mm256_add_epi32(z_rounded_v, zp);
  __m256i w_v = _mm256_add_epi32(w_rounded_v, zp);

  /* Pack to int16_t and saturate */
  __m256i xy_packed_v = _mm256_packs_epi32(x_v, y_v);
  __m256i zw_packed_v = _mm256_packs_epi32(z_v, w_v);

  __m256i xyzw_clamped_v =
      pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

  /*
   * xyzw_clamped_v has results in the following layout so we need to
   * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
   */
  xyzw_clamped_v = _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
  return xyzw_clamped_v;
}
#endif

template<>
struct Vec256<c10::qint8> : public Vec256qi {
    static constexpr int size() {
        return 32;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vec256<float>, 4>;
    using int_vec_return_type = std::array<Vec256<c10::qint32>, 4>;
    using value_type = typename c10::qint8::underlying;

 public:
    using Vec256qi::Vec256qi;

    Vec256() {}
    Vec256(__m256i vals_) { vals = vals_;}

    // Broadcast constructor
    Vec256(const c10::qint8& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi8(uw);
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    // NOLINTNEXTLINE(clang-diagnostic-deprecated-copy)
    Vec256(const Vec256<c10::qint8>& other) : Vec256qi(other.vals) { }

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    static Vec256<c10::qint8> loadu(const void* ptr) {
        return Vec256<c10::qint8>(ptr);
    }

 private:
    __m256i cvtepi8_epi32(__m128i epi8_vals) const {
#ifdef CPU_CAPABILITY_AVX2
        return _mm256_cvtepi8_epi32(epi8_vals);
#else  // CPU_CAPABILITY_AVX2
        __m128i result_data[2];
        __m128i unpacked1 = _mm_unpacklo_epi8(epi8_vals, epi8_vals);
        __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, unpacked1);
        __m128i shifted1 = _mm_srli_si128(epi8_vals, 4);
        __m128i shifted2 = _mm_srai_epi32(unpacked2, 24);
        result_data[0] = shifted2;
        __m128i unpacked3 = _mm_unpacklo_epi8(shifted1, shifted1);
        __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, unpacked3);
        __m128i shifted3 = _mm_srai_epi32(unpacked4, 24);
        result_data[1] = shifted3;
        return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
#endif
    }

 public:
  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_neg_zp_premul) const {
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val3));

#if defined(CPU_CAPABILITY_AVX2)
    auto val0 =
        vec256::fmadd(scale, Vec256<float>(float_val0), scale_neg_zp_premul);
    auto val1 =
        vec256::fmadd(scale, Vec256<float>(float_val1), scale_neg_zp_premul);
    auto val2 =
        vec256::fmadd(scale, Vec256<float>(float_val2), scale_neg_zp_premul);
    auto val3 =
        vec256::fmadd(scale, Vec256<float>(float_val3), scale_neg_zp_premul);
#else
    auto val0 = scale * (Vec256<float>(float_val0) - zero_point);
    auto val1 = scale * (Vec256<float>(float_val1) - zero_point);
    auto val2 = scale * (Vec256<float>(float_val2) - zero_point);
    auto val3 = scale * (Vec256<float>(float_val3) - zero_point);
#endif
    return {val0, val1, val2, val3};
  }

  static Vec256<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    int8_t quantized_values[32];
    QuantizeAvx2<c10::qint8>(
        rhs_data, quantized_values, 32, inverse_scale, zero_point);
    return Vec256<c10::qint8>::loadu(quantized_values);
  }

  Vec256<c10::qint8> maximum(Vec256<c10::qint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_max_epi8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int8_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<int8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<int8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

  Vec256<c10::qint8> minimum(Vec256<c10::qint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epi8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int8_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<int8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::qint8> relu(Vec256<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::qint8> relu6(
        Vec256<c10::qint8> zero_point,
        Vec256<c10::qint8> q_six) {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epi8(
          _mm256_max_epi8(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<int8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<int8_t, size()> zero_point_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      std::array<int8_t, size()> q_six_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      std::array<int8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int8_t>(
            std::max<int8_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    int_vec_return_type widening_subtract(Vec256<c10::qint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

      __m256i int32_val0 = cvtepi8_epi32(int_val0);
      __m256i int32_val1 = cvtepi8_epi32(int_val1);
      __m256i int32_val2 = cvtepi8_epi32(int_val2);
      __m256i int32_val3 = cvtepi8_epi32(int_val3);

      __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
      __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
      __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
      __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

      __m256i int32_b0 = cvtepi8_epi32(int_b0);
      __m256i int32_b1 = cvtepi8_epi32(int_b1);
      __m256i int32_b2 = cvtepi8_epi32(int_b2);
      __m256i int32_b3 = cvtepi8_epi32(int_b3);

      __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
      __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
      __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
      __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);

      return {Vec256<c10::qint32>(res_0),
              Vec256<c10::qint32>(res_1),
              Vec256<c10::qint32>(res_2),
              Vec256<c10::qint32>(res_3)};
#else
      // Pray the compiler can autovectorize this
      std::array<int8_t, size()> int_vals;
      store(int_vals.data());
      std::array<int8_t, size()> b_vals;
      b.store(b_vals.data());
      constexpr int elem_per_int_vec = size() / int_num_vecs();
      int32_t rv[int_num_vecs()][elem_per_int_vec];
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        for (size_t j = 0; j < elem_per_int_vec; ++j) {
          rv[i][j] = static_cast<int32_t>(int_vals[i * elem_per_int_vec + j]) -
              static_cast<int32_t>(b_vals[i * elem_per_int_vec + j]);
        }
      }
      return {Vec256<c10::qint32>::loadu(rv[0]),
              Vec256<c10::qint32>::loadu(rv[1]),
              Vec256<c10::qint32>::loadu(rv[2]),
              Vec256<c10::qint32>::loadu(rv[3])};
#endif
    }

    static Vec256<c10::qint8> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
#ifdef CPU_CAPABILITY_AVX2
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);
      return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
#else
      // Pray the compiler can autovectorize this
      constexpr int elem_per_int_vec = size() / int_num_vecs();
      constexpr auto min_val = std::numeric_limits<value_type>::min();
      constexpr auto max_val = std::numeric_limits<value_type>::max();
      int32_t rv[int_num_vecs()][elem_per_int_vec];
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        inp[i].store(rv[i]);
      }
      std::array<int8_t, size()> result_vals;
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        for (size_t j = 0; j < elem_per_int_vec; ++j) {
          int32_t rounded =
              nearbyint(static_cast<float>(rv[i][j]) * multiplier) + zero_point;
          result_vals[i * elem_per_int_vec + j] =
              std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
        }
      }
      return loadu(result_vals.data());
#endif
    }

    void dump() const {
        for (size_t i = 0; i < size(); ++i) {
            std::cout << (int)((value_type*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    // Load from memory constructor
    Vec256(const void* ptr) {
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::qint8> inline maximum(const Vec256<c10::qint8>& a, const Vec256<c10::qint8>& b) {
  return a.maximum(b);
}

template<>
struct Vec256<c10::quint8> : public Vec256qi {
    static constexpr int size() {
        return 32;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vec256<float>, 4>;
    using int_vec_return_type = std::array<Vec256<c10::qint32>, 4>;
    using value_type = typename c10::quint8::underlying;

 public:
    using Vec256qi::Vec256qi;
    Vec256() {}

    Vec256(__m256i vals_) { vals = vals_;}

    // Broadcast constructor
    Vec256(const c10::quint8& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi8(uw);
    }

    // NOLINTNEXTLINE(clang-diagnostic-deprecated-copy)
    Vec256(const Vec256<c10::quint8>& other) : Vec256qi(other.vals) { }

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    static Vec256<c10::quint8> loadu(const void* ptr) {
        return Vec256<c10::quint8>(ptr);
    }

 private:
    __m256i cvtepu8_epi32(__m128i epu8_vals) const {
#ifdef CPU_CAPABILITY_AVX2
        return _mm256_cvtepu8_epi32(epu8_vals);
#else  // CPU_CAPABILITY_AVX2
        __m128i result_data[2];
        __m128i zeros = _mm_setzero_si128();
        __m128i unpacked1 = _mm_unpacklo_epi8(epu8_vals, zeros);
        __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, zeros);
        result_data[0] = unpacked2;
        __m128i shifted = _mm_srli_si128(epu8_vals, 4);
        __m128i unpacked3 = _mm_unpacklo_epi8(shifted, zeros);
        __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, zeros);
        result_data[1] = unpacked4;
        return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
#endif
    }

 public:
  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val0));
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val3));

#if defined(CPU_CAPABILITY_AVX2)
    auto val0 =
        vec256::fmadd(scale, Vec256<float>(float_val0), scale_zp_premul);
    auto val1 =
        vec256::fmadd(scale, Vec256<float>(float_val1), scale_zp_premul);
    auto val2 =
        vec256::fmadd(scale, Vec256<float>(float_val2), scale_zp_premul);
    auto val3 =
        vec256::fmadd(scale, Vec256<float>(float_val3), scale_zp_premul);
#else
    auto val0 = scale * (Vec256<float>(float_val0) - zero_point);
    auto val1 = scale * (Vec256<float>(float_val1) - zero_point);
    auto val2 = scale * (Vec256<float>(float_val2) - zero_point);
    auto val3 = scale * (Vec256<float>(float_val3) - zero_point);
#endif
    return {val0, val1, val2, val3};
  }

  static Vec256<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    uint8_t quantized_values[32];
    QuantizeAvx2<c10::quint8>(
        rhs_data, quantized_values, 32, inverse_scale, zero_point);
    return Vec256<c10::quint8>::loadu(quantized_values);
  }

  Vec256<c10::quint8> maximum(Vec256<c10::quint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_max_epu8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<uint8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<uint8_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<uint8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<uint8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

  Vec256<c10::quint8> minimum(Vec256<c10::quint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epu8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<uint8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<uint8_t, size()> b_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      std::array<uint8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<uint8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::quint8> relu(Vec256<c10::quint8> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::quint8> relu6(
        Vec256<c10::quint8> zero_point,
        Vec256<c10::quint8> q_six) {
#ifdef CPU_CAPABILITY_AVX2
      return _mm256_min_epu8(
          _mm256_max_epu8(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      std::array<uint8_t, size()> int_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      std::array<uint8_t, size()> zero_point_vals;
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      std::array<uint8_t, size()> q_six_vals;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      std::array<uint8_t, size()> result_vals;
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<uint8_t>(
            std::max<uint8_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    int_vec_return_type widening_subtract(Vec256<c10::quint8> b) const {
#ifdef CPU_CAPABILITY_AVX2
      __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
      __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
      __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
      __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

      __m256i int32_val0 = cvtepu8_epi32(int_val0);
      __m256i int32_val1 = cvtepu8_epi32(int_val1);
      __m256i int32_val2 = cvtepu8_epi32(int_val2);
      __m256i int32_val3 = cvtepu8_epi32(int_val3);

      __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
      __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
      __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
      __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

      __m256i int32_b0 = cvtepu8_epi32(int_b0);
      __m256i int32_b1 = cvtepu8_epi32(int_b1);
      __m256i int32_b2 = cvtepu8_epi32(int_b2);
      __m256i int32_b3 = cvtepu8_epi32(int_b3);

      __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
      __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
      __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
      __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);
      return {Vec256<c10::qint32>(res_0),
              Vec256<c10::qint32>(res_1),
              Vec256<c10::qint32>(res_2),
              Vec256<c10::qint32>(res_3)};
#else
      // Pray the compiler can autovectorize this
      std::array<uint8_t, size()> int_vals;
      std::array<uint8_t, size()> b_vals;
      store(int_vals.data());
      b.store(b_vals.data());
      static constexpr int elem_per_int_vec = size() / int_num_vecs();
      int32_t rv[int_num_vecs()][elem_per_int_vec];
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        for (size_t j = 0; j < elem_per_int_vec; ++j) {
          rv[i][j] = static_cast<int32_t>(int_vals[i * elem_per_int_vec + j]) -
              static_cast<int32_t>(b_vals[i * elem_per_int_vec + j]);
        }
      }
      return {Vec256<c10::qint32>::loadu(rv[0]),
              Vec256<c10::qint32>::loadu(rv[1]),
              Vec256<c10::qint32>::loadu(rv[2]),
              Vec256<c10::qint32>::loadu(rv[3])};
#endif
    }

    static Vec256<c10::quint8> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
#ifdef CPU_CAPABILITY_AVX2
      __m256 multiplier_v = _mm256_set1_ps(multiplier);
      __m256i zero_point_v = _mm256_set1_epi32(zero_point);
      return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
#else
      // Pray the compiler can autovectorize this
      constexpr int elem_per_int_vec = size() / int_num_vecs();
      constexpr auto min_val = std::numeric_limits<value_type>::min();
      constexpr auto max_val = std::numeric_limits<value_type>::max();
      int32_t rv[int_num_vecs()][elem_per_int_vec];
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        inp[i].store(rv[i]);
      }
      std::array<uint8_t, size()> result_vals;
      for (size_t i = 0; i < int_num_vecs(); ++i) {
        for (size_t j = 0; j < elem_per_int_vec; ++j) {
          int32_t rounded =
              nearbyint(static_cast<float>(rv[i][j]) * multiplier) + zero_point;
          result_vals[i * elem_per_int_vec + j] =
              std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
        }
      }
      return loadu(result_vals.data());
#endif
    }

    void dump() const {
        for (size_t i = 0; i < size(); ++i) {
            std::cout << (int)((value_type*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:

    // Load from memory constructor
    Vec256(const void* ptr) {
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::quint8> inline maximum(const Vec256<c10::quint8>& a, const Vec256<c10::quint8>& b) {
  return a.maximum(b);
}

#else

// NOTE: These are low-performance implementations that we fall back on
// if we are not building with AVX2. This may not be an issue, because
// currently for quantization we assume the user has at least AVX512
// installed, so these can simply act as a reference implementation.
//
// If in the future we relax this requirement (AVX2+), we should probably
// revisit these implementations

template <
    typename T,
    typename float_vec_return_type_,
    typename int_vec_return_type_,
    int size_>
struct Vec256QuantizedConverter {
  static constexpr int size() {
    return size_;
  }

  static constexpr int float_num_vecs() {
    return size() / 8;
  }

  static constexpr int int_num_vecs() {
    return size() / 8;
  }

  using float_vec_return_type = float_vec_return_type_;
  using int_vec_return_type = int_vec_return_type_;

  using value_type = typename T::underlying;
  std::array<value_type, size_> vals;

  Vec256QuantizedConverter(T val) {
    for (size_t i = 0; i < size(); ++i) {
      vals[i] = val.val_;
    }
  }

  Vec256QuantizedConverter(const void* ptr) {
    memcpy(vals.data(), ptr, sizeof(value_type) * size());
  }

  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals.data(), count * sizeof(value_type));
  }

  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    float_vec_return_type rv;
    for (int i = 0; i < float_num_vecs(); ++i) {
      float tmp_vals[8];
      for (int j = 0; j < 8; ++j) {
        tmp_vals[j] = at::native::dequantize_val<T>(
            scale[j], zero_point[j], T(vals[8 * i + j]));
      }
      rv[i] = Vec256<float>(tmp_vals[0],
          tmp_vals[1],
          tmp_vals[2],
          tmp_vals[3],
          tmp_vals[4],
          tmp_vals[5],
          tmp_vals[6],
          tmp_vals[7]);
    }
    return rv;
  }

  void dump() const {
      for (int i = 0; i < size(); ++i) {
          std::cout << vals[i] << " ";
      }
      std::cout << std::endl;
  }

 protected:
  Vec256QuantizedConverter() {}
};

template <>
struct Vec256<c10::qint32> : public Vec256QuantizedConverter<
                                 c10::qint32,
                                 std::array<Vec256<float>, 1>,
                                 std::array<Vec256<c10::qint32>, 1>,
                                 8> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            8>() {}
  Vec256(c10::qint32 val)
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            8>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            8>(ptr) {}

  static Vec256<c10::qint32> loadu(const void* ptr) {
    return Vec256<c10::qint32>(ptr);
  }

  static Vec256<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    at::native::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint32*)qvals.data(),
        8 * float_num_vecs());

    return Vec256<c10::qint32>::loadu(qvals.data());
  }

  Vec256<c10::qint32> maximum(Vec256<c10::qint32> b) const {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint32> minimum(Vec256<c10::qint32> b) const {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const  {
    return maximum(zero_point);
  }


  Vec256<c10::qint32> relu6(
      Vec256<c10::qint32> zero_point,
      Vec256<c10::qint32> q_six) {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::qint32> b) const {
    int_vec_return_type retval;
    for (size_t i = 0; i < size(); ++i) {
      retval[0].vals[i] = vals[i] - b.vals[i];
    }
    return retval;
  }

  static Vec256<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] =
          nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
          zero_point;
    }
    return retval;
  }
};

template <>
Vec256<c10::qint32> inline maximum(const Vec256<c10::qint32>& a, const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vec256<c10::qint32> inline operator*(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  Vec256<c10::qint32> retval;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    retval.vals[i] = a.vals[i] * b.vals[i];
  }
  return retval;
}

template <>
Vec256<c10::qint32> inline operator+(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  Vec256<c10::qint32> retval;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    retval.vals[i] = a.vals[i] + b.vals[i];
  }
  return retval;
}

template <>
struct Vec256<c10::qint8> : public Vec256QuantizedConverter<
                                c10::qint8,
                                std::array<Vec256<float>, 4>,
                                std::array<Vec256<c10::qint32>, 4>,
                                32> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>() {}
  Vec256(c10::qint8 val)
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>(ptr) {}

  static Vec256<c10::qint8> loadu(const void* ptr) {
    return Vec256<c10::qint8>(ptr);
  }

  static Vec256<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    at::native::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint8*)qvals.data(),
        8 * float_num_vecs());

    return Vec256<c10::qint8>::loadu(qvals.data());
  }

  Vec256<c10::qint8> maximum(Vec256<c10::qint8> b) const {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint8> minimum(Vec256<c10::qint8> b) const {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint8> relu(Vec256<c10::qint8> zero_point) const {
    return maximum(zero_point);
  }

  Vec256<c10::qint8> relu6(
      Vec256<c10::qint8> zero_point,
      Vec256<c10::qint8> q_six) {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::qint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vec256<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        int32_t rounded =
            nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vec256<c10::qint8> inline maximum(const Vec256<c10::qint8>& a, const Vec256<c10::qint8>& b) {
  return a.maximum(b);
}

template <>
struct Vec256<c10::quint8> : public Vec256QuantizedConverter<
                                 c10::quint8,
                                 std::array<Vec256<float>, 4>,
                                 std::array<Vec256<c10::qint32>, 4>,
                                 32> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>() {}
  Vec256(c10::quint8 val)
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            32>(ptr) {}

  static Vec256<c10::quint8> loadu(const void* ptr) {
    return Vec256<c10::quint8>(ptr);
  }

  static Vec256<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 8> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * 8], 8);
    }

    at::native::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::quint8*)qvals.data(),
        8 * float_num_vecs());

    return Vec256<c10::quint8>::loadu(qvals.data());
  }

  Vec256<c10::quint8> maximum(Vec256<c10::quint8> b) const {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::quint8> minimum(Vec256<c10::quint8> b) const {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::quint8> relu(Vec256<c10::quint8> zero_point) const {
    return maximum(zero_point);
  }


  Vec256<c10::quint8> relu6(
      Vec256<c10::quint8> zero_point,
      Vec256<c10::quint8> q_six) {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::quint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vec256<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        int32_t rounded =
            nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vec256<c10::quint8> inline maximum(const Vec256<c10::quint8>& a, const Vec256<c10::quint8>& b) {
  return a.maximum(b);
}

#endif // (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

}}}
