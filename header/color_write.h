#pragma once
#include "general.h"
#include "intervals.h"
#include <cstdint>
inline Interval4 intv(0.0f,1.0f);

inline vec4 clamp_zero_one(vec4 v){    
    return intv.clamp(v);
}

// Vectorized ACES Filmic Tone Mapping (Narkowicz Fit)
inline vec4 aces_narkowicz_simd4(const vec4& v) {
    vec4 a = _mm_set1_ps(2.51f);
    vec4 b = _mm_set1_ps(0.03f);
    vec4 c = _mm_set1_ps(2.43f);
    vec4 d = _mm_set1_ps(0.59f);
    vec4 e = _mm_set1_ps(0.14f);
    vec4 num = _mm_mul_ps(v, _mm_add_ps(_mm_mul_ps(a, v), b));
    vec4 den = _mm_add_ps(_mm_mul_ps(v, _mm_add_ps(_mm_mul_ps(c, v), d)), e);
    vec4 res = _mm_div_ps(num, den);
    return clamp_zero_one(res);
}

// Vectorized Linear-to-sRGB Conversion
inline vec4 linear_to_srgb_simd4(const vec4& x) {
    vec4 threshold    = _mm_set1_ps(0.0031308f);
    vec4 linear_slope = _mm_set1_ps(12.92f);
    vec4 low_part     = _mm_mul_ps(x, linear_slope);
    vec4 sqrt_x = _mm_sqrt_ps(_mm_max_ps(x, _mm_setzero_ps()));
    vec4 x2     = _mm_mul_ps(x, x);
    vec4 x3     = _mm_mul_ps(x2, x);
    vec4 pow_approx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.662002f), sqrt_x),_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.684122f), x),_mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(-0.323584f), x2),_mm_mul_ps(_mm_set1_ps(0.022541f), x3))));
    vec4 high_part = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(1.055f), pow_approx), _mm_set1_ps(0.055f));
    vec4 is_low    = _mm_cmple_ps(x, threshold);
    vec4 result= _mm_blendv_ps(high_part, low_part, is_low);
    return clamp_zero_one(result);
}



inline void write_color_simd4(vec4 r, vec4 g, vec4 b, float scale, uint32_t out_pixels[4]) {
    vec4 v_scale = _mm_set1_ps(scale);

    // 1. Scale the color values
    r = _mm_mul_ps(r, v_scale);
    g = _mm_mul_ps(g, v_scale);
    b = _mm_mul_ps(b, v_scale);

    // 2. ACES Filmic Tone Mapping
    r = aces_narkowicz_simd4(r);
    g = aces_narkowicz_simd4(g);
    b = aces_narkowicz_simd4(b);

    // 3. Linear to sRGB Transformation
    r = linear_to_srgb_simd4(r);
    g = linear_to_srgb_simd4(g);
    b = linear_to_srgb_simd4(b);

    // 4. Convert float to 32-bit int [0..255]
    vec4 byte_scale = _mm_set1_ps(255.999f);
    vec4i ir = _mm_cvttps_epi32(_mm_mul_ps(r, byte_scale));
    vec4i ig = _mm_cvttps_epi32(_mm_mul_ps(g, byte_scale));
    vec4i ib = _mm_cvttps_epi32(_mm_mul_ps(b, byte_scale));

    alignas(16) int32_t r_arr[4], g_arr[4], b_arr[4];
    _mm_store_si128(reinterpret_cast<vec4i*>(r_arr), ir);
    _mm_store_si128(reinterpret_cast<vec4i*>(g_arr), ig);
    _mm_store_si128(reinterpret_cast<vec4i*>(b_arr), ib);

    // 5. Pack into 32-bit RGBA (0xAABBGGRR / memory byte layout [R, G, B, A])
    for (int k = 0; k < 4; ++k) {
        out_pixels[k] = (255u << 24) |(static_cast<uint8_t>(b_arr[k]) << 16) |(static_cast<uint8_t>(g_arr[k]) << 8)  | static_cast<uint8_t>(r_arr[k]);
    }
}
