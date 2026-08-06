#pragma once
#include "general.h"
#include "intervals.h"

inline void write_color_simd4(__m128 r, __m128 g, __m128 b, float scale, uint32_t out_pixels[4]) {
    __m128 v_scale   = _mm_set1_ps(scale);
    __m128 scale_255 = _mm_set1_ps(256.0f);
    // 1. Scale raw accumulated color values
    r = _mm_mul_ps(r, v_scale);
    g = _mm_mul_ps(g, v_scale);
    b = _mm_mul_ps(b, v_scale);
    // 2. clamping
    static const Interval4 intensity(0.000f, 0.999f);
    r = intensity.clamp(r);
    g = intensity.clamp(g);
    b = intensity.clamp(b);
    // 3. Gamma 2 Correction (sqrt)
    r = _mm_sqrt_ps(r);
    g = _mm_sqrt_ps(g);
    b = _mm_sqrt_ps(b);
    // 4. Convert float [0.0, 1.0] -> int32 [0, 255]
    __m128i ir = _mm_cvttps_epi32(_mm_mul_ps(r, scale_255));
    __m128i ig = _mm_cvttps_epi32(_mm_mul_ps(g, scale_255));
    __m128i ib = _mm_cvttps_epi32(_mm_mul_ps(b, scale_255));

    alignas(16) int32_t r_arr[4], g_arr[4], b_arr[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(r_arr), ir);
    _mm_store_si128(reinterpret_cast<__m128i*>(g_arr), ig);
    _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), ib);
    // 5. Pack into 32-bit RGBA (0xAABBGGRR)
    for (int k = 0; k < 4; ++k) {
        out_pixels[k] = (255u << 24) | (static_cast<uint8_t>(b_arr[k]) << 16) | (static_cast<uint8_t>(g_arr[k]) << 8)  | static_cast<uint8_t>(r_arr[k]);
    }
}
    