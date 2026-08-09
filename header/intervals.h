#pragma once
#include "general.h"


struct alignas(16) Interval4 {
    vec4 min;
    vec4 max;
    Interval4(): min(_mm_set1_ps(+infinity)), max(_mm_set1_ps(-infinity)) {}

    Interval4(float a, float b): min(_mm_set1_ps(a)), max(_mm_set1_ps(b)) {}
    Interval4(vec4 a, vec4 b): min(a), max(b) {}
    Interval4(const Interval4& a, const Interval4& b) {
        min = _mm_min_ps(a.min, b.min);
        max = _mm_max_ps(a.max, b.max);
    }

    inline vec4 length() const {return _mm_sub_ps(max, min);}
    inline vec4 contains(vec4 x) const {
        vec4 ge_min = _mm_cmpge_ps(x, min);
        vec4 le_max = _mm_cmple_ps(x, max);
        return _mm_and_ps(ge_min, le_max);
    }
    inline vec4 surrounds(vec4 x) const {
        vec4 gt_min = _mm_cmpgt_ps(x, min);
        vec4 lt_max = _mm_cmplt_ps(x, max);
        return _mm_and_ps(gt_min, lt_max);
    }
    inline vec4 clamp(vec4 x) const {
        return _mm_max_ps(min, _mm_min_ps(max, x));
    }
    inline Interval4 expand(float delta) const {
        vec4 padding = _mm_set1_ps(delta * 0.5f);
        return Interval4(_mm_sub_ps(min, padding), _mm_add_ps(max, padding));
    }
    // Check if ALL 4 intervals in the packet are degenerate/invalid
    inline bool is_empty_all() const {
        vec4 invalid = _mm_cmpgt_ps(min, max);
        return _mm_movemask_ps(invalid) == 0xF; 
    }
};

// Vectorised addition with scalar offset
inline Interval4 operator+(const Interval4& ival, float displacement) {
    vec4 disp = _mm_set1_ps(displacement);
    return Interval4(_mm_add_ps(ival.min, disp), _mm_add_ps(ival.max, disp));
}