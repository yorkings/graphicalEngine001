#pragma once
#include<mdspan>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <ctime>

#include <omp.h>
#include <chrono>

#include<algorithm>
#include <random> 
#include<cmath>
#include<cstdlib>

#include <atomic>
#include<vector>
#include <sstream>

using std::make_shared;
using std::shared_ptr;

std::string get_time_current(){
    auto now =std::chrono::system_clock::now();
    std::time_t now_time= std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    std::ostringstream oss;
    oss<<std::put_time(local_time,"%Y_%m_%d_%H_%M_%S");
    return oss.str();
}

const float infinity =std::numeric_limits<float>::infinity();
const float pi = M_PI; 
inline float degrees_to_radians(float theta){
    return theta*(pi/180.0f);
}

inline void sincos_simd4(__m128 x, __m128 &out_sin, __m128 &out_cos) {
    // 1. Constants
    const __m128 inv_half_pi = _mm_set1_ps(0.63661977236758134308f); // 2 / pi
    const __m128 half_pi_1   = _mm_set1_ps(1.57079632679489661923f); // pi / 2    
    // Minimax polynomial coefficients for sin(y) and cos(y) on [-pi/4, pi/4]
    const __m128 s1 = _mm_set1_ps(-0.16666666666666666f); // -1/6
    const __m128 s2 = _mm_set1_ps( 0.00833333333333333f); // 1/120
    const __m128 s3 = _mm_set1_ps(-0.000198412698412698f); // -1/5040

    const __m128 c1 = _mm_set1_ps(-0.5f);                  // -1/2
    const __m128 c2 = _mm_set1_ps( 0.04166666666666666f); // 1/24
    const __m128 c3 = _mm_set1_ps(-0.00138888888888888f); // -1/720
    // 2. Single Range Reduction: k = round(x / (pi/2))
    __m128i k_int = _mm_cvtps_epi32(_mm_mul_ps(x, inv_half_pi));
    __m128 k = _mm_cvtepi32_ps(k_int);
    __m128 y = _mm_sub_ps(x, _mm_mul_ps(k, half_pi_1)); // y in [-pi/4, pi/4]
    // 3. Shared y^2 calculation
    __m128 y2 = _mm_mul_ps(y, y);
    // 4. Compute sin(y) polynomial: y + s1*y^3 + s2*y^5 + s3*y^7
    __m128 p_sin = _mm_add_ps(_mm_mul_ps(s3, y2), s2);
    p_sin        = _mm_add_ps(_mm_mul_ps(p_sin, y2), s1);
    p_sin        = _mm_add_ps(_mm_mul_ps(p_sin, y2), _mm_set1_ps(1.0f));
    __m128 sin_y = _mm_mul_ps(y, p_sin);
    // 5. Compute cos(y) polynomial: 1 + c1*y^2 + c2*y^4 + c3*y^6
    __m128 p_cos = _mm_add_ps(_mm_mul_ps(c3, y2), c2);
    p_cos        = _mm_add_ps(_mm_mul_ps(p_cos, y2), c1);
    p_cos        = _mm_add_ps(_mm_mul_ps(p_cos, y2), _mm_set1_ps(1.0f));
    __m128 cos_y = p_cos;
    // 6. Quadrant Mapping (k mod 4)
    __m128i swap_mask_int = _mm_slli_epi32(_mm_and_si128(k_int, _mm_set1_epi32(1)), 31);
    __m128 swap_mask = _mm_castsi128_ps(swap_mask_int);
    // Swap sin_y and cos_y when in odd quadrants (1, 3)
    __m128 sin_val = _mm_or_ps(_mm_and_ps(swap_mask, cos_y), _mm_andnot_ps(swap_mask, sin_y));
    __m128 cos_val = _mm_or_ps(_mm_and_ps(swap_mask, sin_y), _mm_andnot_ps(swap_mask, cos_y));
    // Determine sign bit flips based on k
    // sin sign flip if k=2 or k=3 (bit 1 of k is set)
    __m128i sin_sign_int = _mm_slli_epi32(_mm_and_si128(k_int, _mm_set1_epi32(2)), 30);
    __m128 sin_sign = _mm_castsi128_ps(sin_sign_int);
    // cos sign flip if k=1 or k=2 ( (k+1) bit 1 is set )
    __m128i k_plus_1 = _mm_add_epi32(k_int, _mm_set1_epi32(1));
    __m128i cos_sign_int = _mm_slli_epi32(_mm_and_si128(k_plus_1, _mm_set1_epi32(2)), 30);
    __m128 cos_sign = _mm_castsi128_ps(cos_sign_int);
    out_sin = _mm_xor_ps(sin_val, sin_sign);
    out_cos = _mm_xor_ps(cos_val, cos_sign);
}

inline void degrees_to_radians_simd4(__m128 degrees,__m128 &radians){    
    radians = _mm_mul_ps(degrees,_mm_set1_ps(pi/180.0f));

}

thread_local inline __m128i rng_state_simd4 = _mm_set_epi32(2463534242u, 123456789u, 362436069u, 521288629u);

inline void random_float_simd4(__m128& out_rand) {
    // 1. Advance 4 parallel Xorshift32 generators
    __m128i x = rng_state_simd4;
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
    x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
    rng_state_simd4 = x;
    // 2. Extract 23 bits for IEEE-754 mantissa (0x007FFFFF)
    __m128i mantissa = _mm_and_si128(x, _mm_set1_epi32(0x007FFFFF));
    // 3. Inject exponent bits for float 1.0f (0x3F800000) -> range becomes [1.0f, 2.0f)
    __m128i float_bits = _mm_or_si128(mantissa, _mm_set1_epi32(0x3F800000));
    // 4. Cast to float and subtract 1.0f -> yields range [0.0f, 1.0f)
    __m128 val_1_to_2 = _mm_castsi128_ps(float_bits);
    out_rand = _mm_sub_ps(val_1_to_2, _mm_set1_ps(1.0f));
}
inline void random_float_simd4(__m128& out_rand, float min, float max) {
    __m128 rand_0_to_1;
    random_float_simd4(rand_0_to_1);
    __m128 range = _mm_set1_ps(max - min);
    __m128 min_vec = _mm_set1_ps(min);
    out_rand = _mm_add_ps(_mm_mul_ps(rand_0_to_1, range), min_vec);
}

inline void seed_rng_simd4_quad(int i, int j, int sample_idx) {
    uint32_t p0 = (i + 0) * 1973u + (j + 0) * 9277u + sample_idx * 26699u | 1u;
    uint32_t p1 = (i + 1) * 1973u + (j + 0) * 9277u + sample_idx * 26699u | 1u;
    uint32_t p2 = (i + 0) * 1973u + (j + 1) * 9277u + sample_idx * 26699u | 1u;
    uint32_t p3 = (i + 1) * 1973u + (j + 1) * 9277u + sample_idx * 26699u | 1u;
    rng_state_simd4 = _mm_set_epi32(p3 ^ 0x6C078965u,p2 ^ 0x5D588B65u,p1 ^ 0x41C64E6Du,p0 ^ 0x997F4A7Bu);
}

inline void sin_simd4(__m128 theta, __m128 &sin_theta) {
    __m128 dummy_cos;
    sincos_simd4(theta, sin_theta, dummy_cos);
}

inline void cos_simd4(__m128 theta, __m128 &cos_theta) {
    __m128 dummy_sin;
    sincos_simd4(theta, dummy_sin, cos_theta);
}

inline void tan_simd4(__m128 theta, __m128 &tan_theta) {
    __m128 s, c;
    sincos_simd4(theta, s, c);
    tan_theta = _mm_div_ps(s, c);
}