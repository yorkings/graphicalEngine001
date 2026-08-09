#pragma once
#include "general.h"
#include "intervals.h"
#include <cstdint>

// Vectorized ACES Filmic Tone Mapping (Narkowicz Fit)
inline __m128 aces_narkowicz_simd4(const __m128& v) {
    __m128 a = _mm_set1_ps(2.51f);
    __m128 b = _mm_set1_ps(0.03f);
    __m128 c = _mm_set1_ps(2.43f);
    __m128 d = _mm_set1_ps(0.59f);
    __m128 e = _mm_set1_ps(0.14f);
    __m128 num = _mm_mul_ps(v, _mm_add_ps(_mm_mul_ps(a, v), b));
    __m128 den = _mm_add_ps(_mm_mul_ps(v, _mm_add_ps(_mm_mul_ps(c, v), d)), e);
    __m128 res = _mm_div_ps(num, den);
    return _mm_min_ps(_mm_max_ps(res, _mm_setzero_ps()), _mm_set1_ps(1.0f));
}

// Vectorized Linear-to-sRGB Conversion
inline __m128 linear_to_srgb_simd4(const __m128& x) {
    __m128 threshold    = _mm_set1_ps(0.0031308f);
    __m128 linear_slope = _mm_set1_ps(12.92f);
    __m128 low_part     = _mm_mul_ps(x, linear_slope);
    __m128 sqrt_x = _mm_sqrt_ps(_mm_max_ps(x, _mm_setzero_ps()));
    __m128 x2     = _mm_mul_ps(x, x);
    __m128 x3     = _mm_mul_ps(x2, x);
    __m128 pow_approx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.662002f), sqrt_x),_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.684122f), x),_mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(-0.323584f), x2),_mm_mul_ps(_mm_set1_ps(0.022541f), x3))));
    __m128 high_part = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(1.055f), pow_approx), _mm_set1_ps(0.055f));
    __m128 is_low    = _mm_cmple_ps(x, threshold);
    return _mm_blendv_ps(high_part, low_part, is_low);
}

inline __m128 clamp_zero_one(__m128 v) {
    return _mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
}

inline void write_color_simd4(__m128 r, __m128 g, __m128 b, float scale, uint32_t out_pixels[4]) {
    __m128 v_scale = _mm_set1_ps(scale);

    // 1. Normalize accumulators and sanitize against NaNs/Infs
    r = clamp_zero_one(_mm_mul_ps(r, v_scale));
    g = clamp_zero_one(_mm_mul_ps(g, v_scale));
    b = clamp_zero_one(_mm_mul_ps(b, v_scale));

    // 2. ACES Filmic Tone Mapping
    r = aces_narkowicz_simd4(r);
    g = aces_narkowicz_simd4(g);
    b = aces_narkowicz_simd4(b);

    // 3. Linear to sRGB Transformation
    r = linear_to_srgb_simd4(r);
    g = linear_to_srgb_simd4(g);
    b = linear_to_srgb_simd4(b);

    // 4. Convert float to 32-bit int [0..255]
    __m128 byte_scale = _mm_set1_ps(255.999f);
    __m128i ir = _mm_cvttps_epi32(_mm_mul_ps(r, byte_scale));
    __m128i ig = _mm_cvttps_epi32(_mm_mul_ps(g, byte_scale));
    __m128i ib = _mm_cvttps_epi32(_mm_mul_ps(b, byte_scale));

    alignas(16) int32_t r_arr[4], g_arr[4], b_arr[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(r_arr), ir);
    _mm_store_si128(reinterpret_cast<__m128i*>(g_arr), ig);
    _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), ib);

    // 5. Pack into 32-bit RGBA (0xAABBGGRR / memory byte layout [R, G, B, A])
    for (int k = 0; k < 4; ++k) {
        out_pixels[k] = (255u << 24) |(static_cast<uint8_t>(b_arr[k]) << 16) |(static_cast<uint8_t>(g_arr[k]) << 8)  | static_cast<uint8_t>(r_arr[k]);
    }
}
