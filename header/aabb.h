#pragma once
#include "ray.h"
#include "intervals.h"


struct alignas(16) aabb4 {
    vec4 min_x, min_y, min_z;
    vec4 max_x, max_y, max_z;
    aabb4() {
        min_x = min_y = min_z = _mm_set1_ps(infinity);
        max_x = max_y = max_z = _mm_set1_ps(-infinity);
    }

    aabb4(const vec3& a, const vec3& b) {
        constexpr float eps = 1e-5f;        
        float mx = std::min(a.x(), b.x());
        float Mx = std::max(a.x(), b.x());
        float my = std::min(a.y(), b.y());
        float My = std::max(a.y(), b.y());
        float mz = std::min(a.z(), b.z());
        float Mz = std::max(a.z(), b.z());

        if (Mx - mx < eps) { mx -= eps; Mx += eps; }
        if (My - my < eps) { my -= eps; My += eps; }
        if (Mz - mz < eps) { mz -= eps; Mz += eps; }

        min_x = _mm_set1_ps(mx); min_y = _mm_set1_ps(my); min_z = _mm_set1_ps(mz);
        max_x = _mm_set1_ps(Mx); max_y = _mm_set1_ps(My); max_z = _mm_set1_ps(Mz);
    }


    aabb4(const vec3 min_pts[4], const vec3 max_pts[4], int valid_count = 4) {
        alignas(16) float mx[4], my[4], mz[4];
        alignas(16) float Mx[4], My[4], Mz[4];      

        for (int i = 0; i < 4; ++i) {
            if (i < valid_count) {
                mx[i] = min_pts[i].x(); my[i] = min_pts[i].y(); mz[i] = min_pts[i].z();
                Mx[i] = max_pts[i].x(); My[i] = max_pts[i].y(); Mz[i] = max_pts[i].z();
            } else {
                mx[i] = my[i] = mz[i] = infinity;
                Mx[i] = My[i] = Mz[i] = -infinity;
            }
        }
        min_x = _mm_load_ps(mx); min_y = _mm_load_ps(my); min_z = _mm_load_ps(mz);
        max_x = _mm_load_ps(Mx); max_y = _mm_load_ps(My); max_z = _mm_load_ps(Mz);
    }
    inline aabb4 flatten() const {
        vec4 mx1 = _mm_min_ps(min_x, _mm_shuffle_ps(min_x, min_x, _MM_SHUFFLE(1,0,3,2)));
        vec4 mx2 = _mm_min_ps(mx1,   _mm_shuffle_ps(mx1,   mx1,   _MM_SHUFFLE(0,1,0,1)));

        vec4 my1 = _mm_min_ps(min_y, _mm_shuffle_ps(min_y, min_y, _MM_SHUFFLE(1,0,3,2)));
        vec4 my2 = _mm_min_ps(my1,   _mm_shuffle_ps(my1,   my1,   _MM_SHUFFLE(0,1,0,1)));

        vec4 mz1 = _mm_min_ps(min_z, _mm_shuffle_ps(min_z, min_z, _MM_SHUFFLE(1,0,3,2)));
        vec4 mz2 = _mm_min_ps(mz1,   _mm_shuffle_ps(mz1,   mz1,   _MM_SHUFFLE(0,1,0,1)));

        vec4 Mx1 = _mm_max_ps(max_x, _mm_shuffle_ps(max_x, max_x, _MM_SHUFFLE(1,0,3,2)));
        vec4 Mx2 = _mm_max_ps(Mx1,   _mm_shuffle_ps(Mx1,   Mx1,   _MM_SHUFFLE(0,1,0,1)));

        vec4 My1 = _mm_max_ps(max_y, _mm_shuffle_ps(max_y, max_y, _MM_SHUFFLE(1,0,3,2)));
        vec4 My2 = _mm_max_ps(My1,   _mm_shuffle_ps(My1,   My1,   _MM_SHUFFLE(0,1,0,1)));

        vec4 Mz1 = _mm_max_ps(max_z, _mm_shuffle_ps(max_z, max_z, _MM_SHUFFLE(1,0,3,2)));
        vec4 Mz2 = _mm_max_ps(Mz1,   _mm_shuffle_ps(Mz1,   Mz1,   _MM_SHUFFLE(0,1,0,1)));

        aabb4 res;
        res.min_x = mx2; res.min_y = my2; res.min_z = mz2;
        res.max_x = Mx2; res.max_y = My2; res.max_z = Mz2;
        return res;
    }
    inline static aabb4 combine(const aabb4& b1, const aabb4& b2) {
        aabb4 f1 = b1.flatten();
        aabb4 f2 = b2.flatten();
        aabb4 res;
        res.min_x = _mm_min_ps(f1.min_x, f2.min_x);
        res.min_y = _mm_min_ps(f1.min_y, f2.min_y);
        res.min_z = _mm_min_ps(f1.min_z, f2.min_z);
        res.max_x = _mm_max_ps(f1.max_x, f2.max_x);
        res.max_y = _mm_max_ps(f1.max_y, f2.max_y);
        res.max_z = _mm_max_ps(f1.max_z, f2.max_z);
        return res;
    }
  
    //MODE 1: Test 4 Rays (in a packet) vs 1 Box (BVH Node Traversal)
    inline vec4 hit_packet(const Raypackets& pack, Interval4 ray_t) const {
        vec4 t0_x = _mm_mul_ps(_mm_sub_ps(min_x, pack.orig_x), pack.inv_dir_x);
        vec4 t1_x = _mm_mul_ps(_mm_sub_ps(max_x, pack.orig_x), pack.inv_dir_x);
        vec4 t_min = _mm_max_ps(ray_t.min, _mm_min_ps(t0_x, t1_x));
        vec4 t_max = _mm_min_ps(ray_t.max, _mm_max_ps(t0_x, t1_x));

        vec4 t0_y = _mm_mul_ps(_mm_sub_ps(min_y, pack.orig_y), pack.inv_dir_y);
        vec4 t1_y = _mm_mul_ps(_mm_sub_ps(max_y, pack.orig_y), pack.inv_dir_y);
        t_min = _mm_max_ps(t_min, _mm_min_ps(t0_y, t1_y));
        t_max = _mm_min_ps(t_max, _mm_max_ps(t0_y, t1_y));

        vec4 t0_z = _mm_mul_ps(_mm_sub_ps(min_z, pack.orig_z), pack.inv_dir_z);
        vec4 t1_z = _mm_mul_ps(_mm_sub_ps(max_z, pack.orig_z), pack.inv_dir_z);
        t_min = _mm_max_ps(t_min, _mm_min_ps(t0_z, t1_z));
        t_max = _mm_min_ps(t_max, _mm_max_ps(t0_z, t1_z));

        return _mm_cmple_ps(t_min, t_max);
    }

    // MODE 2: Test 1 Ray (broadcasted across registers) vs 4 Child Boxes (BVH4 Node Traversal)
    inline vec4 hit_1ray_vs_4boxes(vec4 r_ox, vec4 r_oy, vec4 r_oz, vec4 r_inv_dx, vec4 r_inv_dy, vec4 r_inv_dz, vec4 t_min_in, vec4 t_max_in) const {
        vec4 t0_x = _mm_mul_ps(_mm_sub_ps(min_x, r_ox), r_inv_dx);
        vec4 t1_x = _mm_mul_ps(_mm_sub_ps(max_x, r_ox), r_inv_dx);
        vec4 t_min = _mm_max_ps(t_min_in, _mm_min_ps(t0_x, t1_x));
        vec4 t_max = _mm_min_ps(t_max_in, _mm_max_ps(t0_x, t1_x));

        vec4 t0_y = _mm_mul_ps(_mm_sub_ps(min_y, r_oy), r_inv_dy);
        vec4 t1_y = _mm_mul_ps(_mm_sub_ps(max_y, r_oy), r_inv_dy);
        t_min = _mm_max_ps(t_min, _mm_min_ps(t0_y, t1_y));
        t_max = _mm_min_ps(t_max, _mm_max_ps(t0_y, t1_y));

        vec4 t0_z = _mm_mul_ps(_mm_sub_ps(min_z, r_oz), r_inv_dz);
        vec4 t1_z = _mm_mul_ps(_mm_sub_ps(max_z, r_oz), r_inv_dz);
        t_min = _mm_max_ps(t_min, _mm_min_ps(t0_z, t1_z));
        t_max = _mm_min_ps(t_max, _mm_max_ps(t0_z, t1_z));

        return _mm_cmple_ps(t_min, t_max); // 4-bit mask indicating hit status for child boxes 0..3
    }
};