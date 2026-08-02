#pragma once
#include "general.h"

struct alignas(16) Interval4 {
    __m128 min;
    __m128 max;
    Interval4(): min(_mm_set1_ps(+infinity)), max(_mm_set1_ps(-infinity)) {}

    // Broadcast scalar values across all 4 lanes
    Interval4(float a, float b): min(_mm_set1_ps(a)), max(_mm_set1_ps(b)) {}
    // Direct SSE vector initialization
    Interval4(__m128 a, __m128 b): min(a), max(b) {}
    // Create tightly enclosing intervals for 4 pairs simultaneously
    Interval4(const Interval4& a, const Interval4& b) {
        min = _mm_min_ps(a.min, b.min);
        max = _mm_max_ps(a.max, b.max);
    }

    inline __m128 length() const {return _mm_sub_ps(max, min);}
    // Returns bitmask where lane is true if (min <= x <= max)
    inline __m128 contains(__m128 x) const {
        __m128 ge_min = _mm_cmpge_ps(x, min);
        __m128 le_max = _mm_cmple_ps(x, max);
        return _mm_and_ps(ge_min, le_max);
    }
    // Returns bitmask where lane is true if (min < x < max)
    inline __m128 surrounds(__m128 x) const {
        __m128 gt_min = _mm_cmpgt_ps(x, min);
        __m128 lt_max = _mm_cmplt_ps(x, max);
        return _mm_and_ps(gt_min, lt_max);
    }
    inline __m128 clamp(__m128 x) const {
        return _mm_max_ps(min, _mm_min_ps(max, x));
    }
    inline Interval4 expand(float delta) const {
        __m128 padding = _mm_set1_ps(delta * 0.5f);
        return Interval4(_mm_sub_ps(min, padding), _mm_add_ps(max, padding));
    }
    // Check if ALL 4 intervals in the packet are degenerate/invalid
    inline bool is_empty_all() const {
        __m128 invalid = _mm_cmpgt_ps(min, max);
        return _mm_movemask_ps(invalid) == 0xF; 
    }
};

// Vectorised addition with scalar offset
inline Interval4 operator+(const Interval4& ival, float displacement) {
    __m128 disp = _mm_set1_ps(displacement);
    return Interval4(_mm_add_ps(ival.min, disp), _mm_add_ps(ival.max, disp));
}