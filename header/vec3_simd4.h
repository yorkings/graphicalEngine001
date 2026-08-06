#pragma once
#include "vec3.h"
using vec4=__m128;
inline void random_unit_vector_simd4(vec4& out_x, vec4& out_y, vec4& out_z) {
    vec3 r0 = random_unit_vector();
    vec3 r1 = random_unit_vector();
    vec3 r2 = random_unit_vector();
    vec3 r3 = random_unit_vector();
    out_x = _mm_set_ps(r3.x(), r2.x(), r1.x(), r0.x());
    out_y = _mm_set_ps(r3.y(), r2.y(), r1.y(), r0.y());
    out_z = _mm_set_ps(r3.z(), r2.z(), r1.z(), r0.z());
}

inline void reflect_simd4(const vec4& v_x, const vec4& v_y, const vec4& v_z,const vec4& n_x, const vec4& n_y, const vec4& n_z,vec4& out_x, vec4& out_y, vec4& out_z) {
   vec4 dot=_mm_add_ps(_mm_mul_ps(v_x,n_x),_mm_add_ps(_mm_mul_ps(v_y,n_y),_mm_mul_ps(v_z,n_z)));
   vec4 two_dot=_mm_mul_ps(_mm_set1_ps(2.0f),dot);
   out_x=_mm_sub_ps(v_x,_mm_mul_ps(two_dot,n_x));
   out_y=_mm_sub_ps(v_y,_mm_mul_ps(two_dot,n_y));
   out_z=_mm_sub_ps(v_z,_mm_mul_ps(two_dot,n_z));
}

inline void refract_simd4(const vec4& uv_x, const vec4& uv_y, const vec4& uv_z,const vec4& n_x, const vec4& n_y, const vec4& n_z,const __m128& etai_over_etat,
    vec4& out_x, vec4& out_y, vec4& out_z){
    vec4 dot=_mm_add_ps(_mm_mul_ps(uv_x,n_x),_mm_add_ps(_mm_mul_ps(uv_y,n_y),_mm_mul_ps(uv_z,n_z)));
    vec4 neg_dot=_mm_sub_ps(_mm_setzero_ps(),dot);
    vec4 cos_theta=_mm_min_ps(neg_dot,_mm_set1_ps(1.0f));
    //perpendicular output
    vec4 perp_x=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_x,_mm_mul_ps(cos_theta,n_x)));
    vec4 perp_y=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_y,_mm_mul_ps(cos_theta,n_y)));
    vec4 perp_z=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_z,_mm_mul_ps(cos_theta,n_z)));
    //paralell output -sqrt(max(0.0, 1.0 - |r_out_perp|^2)) * n
    vec4 perp_len_sq = _mm_add_ps(_mm_mul_ps(perp_x, perp_x), _mm_add_ps(_mm_mul_ps(perp_y, perp_y), _mm_mul_ps(perp_z, perp_z)));
    vec4 sub = _mm_sub_ps(_mm_set1_ps(1.0f), perp_len_sq);
    vec4 max_sub = _mm_max_ps(_mm_setzero_ps(), sub); // Safe against NaN on TIR
    vec4 parallel_coeff = _mm_sub_ps(_mm_setzero_ps(), _mm_sqrt_ps(max_sub));  

    out_x = _mm_add_ps(perp_x, _mm_mul_ps(parallel_coeff, n_x));
    out_y = _mm_add_ps(perp_y, _mm_mul_ps(parallel_coeff, n_y));
    out_z = _mm_add_ps(perp_z, _mm_mul_ps(parallel_coeff, n_z)); 
}