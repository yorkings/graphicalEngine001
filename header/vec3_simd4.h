#pragma once
#include "vec3.h"

inline void dot_simd4(const vec4& a_x, const vec4& a_y, const vec4& a_z,const vec4& b_x, const vec4& b_y, const vec4& b_z,vec4& out_dot) {
    out_dot = _mm_add_ps(_mm_mul_ps(a_x, b_x), _mm_add_ps(_mm_mul_ps(a_y, b_y), _mm_mul_ps(a_z, b_z)));
}

inline void normalize_simd4(const vec4& x, const vec4& y, const vec4& z,vec4& out_x, vec4& out_y, vec4& out_z) {
    vec4 len_sq;
    dot_simd4(x,y,z,x,y,z,len_sq);
    vec4 safe_len_sq = _mm_max_ps(len_sq, _mm_set1_ps(1e-12f));
    vec4 inv_len = _mm_rsqrt_ps(safe_len_sq);
    out_x = _mm_mul_ps(x, inv_len);
    out_y = _mm_mul_ps(y, inv_len);
    out_z = _mm_mul_ps(z, inv_len);
}



inline void cross_simd4(const vec4& a_x, const vec4& a_y, const vec4& a_z,const vec4& b_x, const vec4& b_y, const vec4& b_z,vec4& out_x, vec4& out_y, vec4& out_z) {
    out_x = _mm_sub_ps(_mm_mul_ps(a_y, b_z), _mm_mul_ps(a_z, b_y));
    out_y = _mm_sub_ps(_mm_mul_ps(a_z, b_x), _mm_mul_ps(a_x, b_z));
    out_z = _mm_sub_ps(_mm_mul_ps(a_x, b_y), _mm_mul_ps(a_y, b_x));
}

inline void random_unit_vector_simd4(vec4& out_x, vec4& out_y, vec4& out_z) {
    vec4 rx, ry, rz,len_sq ;
    vec4 active_mask = _mm_castsi128_ps(_mm_set1_epi32(-1));
    vec4 res_x = _mm_setzero_ps();
    vec4 res_y = _mm_setzero_ps();
    vec4 res_z = _mm_setzero_ps();
    // Rejection loop until all 4 SIMD lanes find a valid point inside unit sphere
    while (_mm_movemask_ps(active_mask) != 0) {
        vec4 u1, u2, u3;
        random_float_simd4(u1);
        random_float_simd4(u2);
        random_float_simd4(u3);
        // Map [0, 1) -> [-1, 1)
        rx = _mm_sub_ps(_mm_mul_ps(u1, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
        ry = _mm_sub_ps(_mm_mul_ps(u2, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
        rz = _mm_sub_ps(_mm_mul_ps(u3, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));

        dot_simd4(rx,ry,rz,rx,ry,rz,len_sq);
        // Valid if 1e-5 < len_sq < 1.0
        vec4 valid_mask = _mm_and_ps(_mm_cmpgt_ps(len_sq, _mm_set1_ps(1e-5f)),_mm_cmplt_ps(len_sq, _mm_set1_ps(1.0f)));
        vec4 store_mask = _mm_and_ps(active_mask, valid_mask);
        // Store valid lane results
        res_x = _mm_blendv_ps(res_x, rx, store_mask);
        res_y = _mm_blendv_ps(res_y, ry, store_mask);
        res_z = _mm_blendv_ps(res_z, rz, store_mask);
        // Deactivate finished lanes
        active_mask = _mm_andnot_ps(store_mask, active_mask);
    }
    // Normalize valid points to get unit length vectors
    normalize_simd4(res_x, res_y, res_z, out_x, out_y, out_z);
}

inline void reflect_simd4(const vec4& v_x, const vec4& v_y, const vec4& v_z,const vec4& n_x, const vec4& n_y, const vec4& n_z,vec4& out_x, vec4& out_y, vec4& out_z) {
   vec4 dot,two_dot;
   dot_simd4(v_x,v_y,v_z,n_x,n_y,n_z,dot);
   two_dot=_mm_mul_ps(_mm_set1_ps(2.0f),dot);
   out_x=_mm_sub_ps(v_x,_mm_mul_ps(two_dot,n_x));
   out_y=_mm_sub_ps(v_y,_mm_mul_ps(two_dot,n_y));
   out_z=_mm_sub_ps(v_z,_mm_mul_ps(two_dot,n_z));
}

inline void refract_simd4(const vec4& uv_x, const vec4& uv_y, const vec4& uv_z,const vec4& n_x, const vec4& n_y, const vec4& n_z,const vec4& cos_theta,const vec4& etai_over_etat,
    vec4& out_x, vec4& out_y, vec4& out_z){
    //perpendicular output
    vec4 perp_x=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_x,_mm_mul_ps(cos_theta,n_x)));
    vec4 perp_y=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_y,_mm_mul_ps(cos_theta,n_y)));
    vec4 perp_z=_mm_mul_ps(etai_over_etat,_mm_add_ps(uv_z,_mm_mul_ps(cos_theta,n_z)));
    //paralell output -sqrt(max(0.0, 1.0 - |r_out_perp|^2)) * n
    vec4 perp_len_sq;
    dot_simd4(perp_x,perp_y,perp_z,perp_x,perp_y,perp_z,perp_len_sq);
    vec4 sub = _mm_sub_ps(_mm_set1_ps(1.0f), perp_len_sq);
    vec4 max_sub = _mm_max_ps(_mm_setzero_ps(), sub); 
    vec4 par_factor = _mm_sub_ps(_mm_setzero_ps(), _mm_sqrt_ps(max_sub));  

    out_x = _mm_add_ps(perp_x, _mm_mul_ps(par_factor, n_x));
    out_y = _mm_add_ps(perp_y, _mm_mul_ps(par_factor, n_y));
    out_z = _mm_add_ps(perp_z, _mm_mul_ps(par_factor, n_z)); 
}




