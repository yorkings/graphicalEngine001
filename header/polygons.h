#pragma once
#include "ray.h"
#include "hit_funcs.h"

class sphere: public hitable{
    private:
        vec4 cen_x,cen_y,cen_z;
        vec4 radius;
        vec4 inv_radius;
        shared_ptr<material> mat;
    public:
        sphere(vec3 center, float r,shared_ptr<material> mat_ptr): mat(mat_ptr){
            cen_x = _mm_set1_ps(center.x());
            cen_y = _mm_set1_ps(center.y());
            cen_z = _mm_set1_ps(center.z());
            radius = _mm_set1_ps(r);
            inv_radius = _mm_set1_ps(1.0f / r);

        }
        sphere(vec4 cx, vec4 cy, vec4 cz, vec4 r,std::shared_ptr<material> mat_ptr): cen_x(cx), cen_y(cy), cen_z(cz), radius(r), mat(mat_ptr.get()){
            inv_radius = _mm_div_ps(_mm_set1_ps(1.0f), r);
        }

        bool hit(const Raypackets &r_packs,const Interval4& ray_t,hit_rec& rec)const{
            vec4 oc_x=_mm_sub_ps(r_packs.orig_x,cen_x);
            vec4 oc_y= _mm_sub_ps(r_packs.orig_y,cen_y);
            vec4 oc_z=_mm_sub_ps(r_packs.orig_z,cen_z);

            vec4 a,half_b,oc_sq;
            dot_simd4(r_packs.dir_x,r_packs.dir_y,r_packs.dir_z,r_packs.dir_x,r_packs.dir_y,r_packs.dir_z,a);
            dot_simd4(r_packs.dir_x,r_packs.dir_y,r_packs.dir_z,oc_x,oc_y,oc_z,half_b);
            dot_simd4(oc_x,oc_y,oc_z,oc_x,oc_y,oc_z,oc_sq);
            vec4 c=_mm_sub_ps(oc_sq,_mm_mul_ps(radius, radius));

            vec4 discriminant = _mm_sub_ps(_mm_mul_ps(half_b, half_b), _mm_mul_ps(a, c));
           
            vec4 has_hit = _mm_cmpge_ps(discriminant, _mm_setzero_ps());
            if (_mm_movemask_ps(has_hit) == 0) {
                return false;
            }
            vec4 safe_disc = _mm_max_ps(discriminant, _mm_setzero_ps());
            vec4 sqrtd = _mm_sqrt_ps(safe_disc);
            // Single reciprocal division replaces two vector divisions
            vec4 inv_a = _mm_div_ps(_mm_set1_ps(1.0f), a);
            vec4 neg_half_b = _mm_sub_ps(_mm_setzero_ps(), half_b);

            vec4 root1 = _mm_mul_ps(_mm_sub_ps(neg_half_b, sqrtd), inv_a);
            vec4 root2 = _mm_mul_ps(_mm_add_ps(neg_half_b, sqrtd), inv_a);           

            vec4 r1_valid = ray_t.surrounds(root1);
            vec4 r2_valid = ray_t.surrounds(root2);

            vec4 chosen_t = _mm_blendv_ps(root2,root1,r1_valid);
            vec4 valid_root_mask = _mm_or_ps(r1_valid, r2_valid);
            vec4 final_hit_mask = _mm_and_ps(has_hit, valid_root_mask);
            if (_mm_movemask_ps(final_hit_mask) == 0) {
                return false;
            }

            rec.hit_mask = final_hit_mask;
            rec.t = chosen_t;
            ray_at_t4(rec.p_x, rec.p_y, rec.p_z, r_packs, rec.t);

            vec4 out_nx = _mm_mul_ps(_mm_sub_ps(rec.p_x, cen_x), inv_radius);
            vec4 out_ny = _mm_mul_ps(_mm_sub_ps(rec.p_y, cen_y), inv_radius);
            vec4 out_nz = _mm_mul_ps(_mm_sub_ps(rec.p_z, cen_z), inv_radius);

            rec.set_face_normal_4(r_packs, out_nx, out_ny, out_nz);
            rec.mat.fill(mat.get());
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