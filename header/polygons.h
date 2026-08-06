#pragma once
#include "ray.h"
#include "hit_funcs.h"

class sphere: public hitable{
    private:
        __m128 cen_x,cen_y,cen_z;
        __m128 radius;
        __m128 inv_radius;
    public:
        sphere(vec3 center, float r) {
            cen_x = _mm_set1_ps(center.x());
            cen_y = _mm_set1_ps(center.y());
            cen_z = _mm_set1_ps(center.z());
            radius = _mm_set1_ps(r);
            inv_radius = _mm_set1_ps(1.0f / r);

        }
        sphere(__m128 cx, __m128 cy, __m128 cz, __m128 r): cen_x(cx), cen_y(cy), cen_z(cz), radius(r){
            inv_radius = _mm_div_ps(_mm_set1_ps(1.0f), r);
        }

        bool hit(const Raypackets &r_packs,const Interval4& ray_t,hit_rec& rec)const{
            //oc object center
            __m128 oc_x=_mm_sub_ps(r_packs.orig_x,cen_x);
            __m128 oc_y= _mm_sub_ps(r_packs.orig_y,cen_y);
            __m128 oc_z=_mm_sub_ps(r_packs.orig_z,cen_z);

            __m128 a = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r_packs.dir_x, r_packs.dir_x), _mm_mul_ps(r_packs.dir_y, r_packs.dir_y)),
            _mm_mul_ps(r_packs.dir_z, r_packs.dir_z));

            __m128 half_b=_mm_add_ps(_mm_add_ps(_mm_mul_ps(oc_x,r_packs.dir_x), _mm_mul_ps(oc_y, r_packs.dir_y)),_mm_mul_ps(oc_z,r_packs.dir_z));

            __m128 oc_sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(oc_x, oc_x), _mm_mul_ps(oc_y, oc_y)), _mm_mul_ps(oc_z, oc_z));
            __m128 c=_mm_sub_ps(oc_sq,_mm_mul_ps(radius, radius));

            __m128 discriminant = _mm_sub_ps(_mm_mul_ps(half_b, half_b), _mm_mul_ps(a, c));

            __m128 has_hit = _mm_cmpge_ps(discriminant, _mm_setzero_ps());
            // Early return if NONE of the 4 rays hit the sphere's bounding math
            if (_mm_movemask_ps(has_hit) == 0) {
                return false;
            }
            __m128 sqrtd = _mm_sqrt_ps(discriminant);
            // root1 = (-half_b - sqrtd) / a
            __m128 root1 = _mm_div_ps(_mm_sub_ps(_mm_setzero_ps(), _mm_add_ps(half_b, sqrtd)), a);
            // root2 = (-half_b + sqrtd) / a
            __m128 root2 = _mm_div_ps(_mm_add_ps(_mm_sub_ps(_mm_setzero_ps(), half_b), sqrtd), a);

            __m128 r1_valid = ray_t.surrounds(root1);
            __m128 r2_valid = ray_t.surrounds(root2);
            __m128 chosen_t = _mm_blendv_ps(root2,root1,r1_valid);
            // A lane is valid ONLY if it hit AND has at least one root in (ray_t.min, ray_t.max)
            __m128 valid_root_mask = _mm_or_ps(r1_valid, r2_valid);
            __m128 final_hit_mask = _mm_and_ps(has_hit, valid_root_mask);
            // Early exit if none of the 4 rays hit within the active interval range
            if (_mm_movemask_ps(final_hit_mask) == 0) {
                return false;
            }
            rec.hit_mask = final_hit_mask;
            rec.t = chosen_t;
            ray_at_t4(rec.p_x, rec.p_y, rec.p_z, r_packs, rec.t);
            // Outward normal = (p - center) / radius
            __m128 out_nx = _mm_mul_ps(_mm_sub_ps(rec.p_x, cen_x), inv_radius);
            __m128 out_ny = _mm_mul_ps(_mm_sub_ps(rec.p_y, cen_y), inv_radius);
            __m128 out_nz = _mm_mul_ps(_mm_sub_ps(rec.p_z, cen_z), inv_radius);

            rec.set_face_normal_4(r_packs, out_nx, out_ny, out_nz);
            return true;
        }
};

// class quad{
    // private:
        // 
    // public:
        // quad(){}
        // bool hit(){}
// };
// 
// class triangle{
    // public:
        // triangle(){}
        // bool hit(){}
// };