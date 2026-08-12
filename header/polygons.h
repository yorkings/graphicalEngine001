#pragma once
#include "ray.h"
#include "hit_funcs.h"

class sphere: public hitable{
    private:
        vec4 cen1_x, cen1_y, cen1_z;
        vec4 vec_x, vec_y, vec_z;        
        vec4 radius;
        vec4 inv_radius;
        shared_ptr<material> mat;
        aabb4 bbox;
        inline __m128 hmin_ps(__m128 v) {
            v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2))); // Compare lane pairs
            return _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1))); // Compare adjacent
        }

        inline __m128 hmax_ps(__m128 v) {
            v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
        }
    public:
        sphere(vec3 center, float r,shared_ptr<material> mat_ptr): mat(mat_ptr){
            cen1_x = _mm_set1_ps(center.x());
            cen1_y = _mm_set1_ps(center.y());
            cen1_z = _mm_set1_ps(center.z());

            vec_x = _mm_setzero_ps();
            vec_y = _mm_setzero_ps();
            vec_z = _mm_setzero_ps();

            radius = _mm_set1_ps(r);
            inv_radius = _mm_set1_ps(1.0f / r);
            vec3 rvec(r, r, r);
            bbox = aabb4(center - rvec, center + rvec);

        }
        sphere(vec4 cx, vec4 cy, vec4 cz, vec4 r, std::shared_ptr<material> mat_ptr): cen1_x(cx), cen1_y(cy), cen1_z(cz), radius(r), mat(mat_ptr),
        vec_x(_mm_setzero_ps()), vec_y(_mm_setzero_ps()), vec_z(_mm_setzero_ps()) {
            inv_radius = _mm_div_ps(_mm_set1_ps(1.0f), r);

            vec4 lane_min_x = _mm_sub_ps(cx, r);
            vec4 lane_max_x = _mm_add_ps(cx, r);
            vec4 lane_min_y = _mm_sub_ps(cy, r);
            vec4 lane_max_y = _mm_add_ps(cy, r);
            vec4 lane_min_z = _mm_sub_ps(cz, r);
            vec4 lane_max_z = _mm_add_ps(cz, r);

            bbox.min_x = hmin_ps(lane_min_x);
            bbox.max_x = hmax_ps(lane_max_x);                
            bbox.min_y = hmin_ps(lane_min_y);
            bbox.max_y = hmax_ps(lane_max_y);
            bbox.min_z = hmin_ps(lane_min_z);
            bbox.max_z = hmax_ps(lane_max_z);
        }
        //moving sphere
        sphere(const vec3& center1, const vec3& center2, float r, std::shared_ptr<material> mat_ptr): mat(mat_ptr) {
            cen1_x = _mm_set1_ps(center1.x());
            cen1_y = _mm_set1_ps(center1.y());
            cen1_z = _mm_set1_ps(center1.z());            
            vec3 velocity = center2 - center1;
            vec_x = _mm_set1_ps(velocity.x());
            vec_y = _mm_set1_ps(velocity.y());
            vec_z = _mm_set1_ps(velocity.z());
            radius = _mm_set1_ps(r);
            inv_radius = _mm_div_ps(_mm_set1_ps(1.0f), _mm_set1_ps(r));
            vec3 rvec(r, r, r);
            aabb4 box1(center1 - rvec, center1 + rvec);
            aabb4 box2(center2 - rvec, center2 + rvec);
            bbox=aabb4::combine(box1,box2);
        }

        aabb4 bounding_box() const override { return bbox; }
        bool hit(const Raypackets &r_packs,const Interval4 ray_t,hit_rec& rec)const{
            // Instantaneous center per lane: C(t) = C1 + t * V
            vec4 cur_cen_x = _mm_add_ps(cen1_x, _mm_mul_ps(r_packs.time, vec_x));
            vec4 cur_cen_y = _mm_add_ps(cen1_y, _mm_mul_ps(r_packs.time, vec_y));
            vec4 cur_cen_z = _mm_add_ps(cen1_z, _mm_mul_ps(r_packs.time, vec_z));

            vec4 oc_x=_mm_sub_ps(r_packs.orig_x,cur_cen_x);
            vec4 oc_y= _mm_sub_ps(r_packs.orig_y,cur_cen_y);
            vec4 oc_z=_mm_sub_ps(r_packs.orig_z,cur_cen_z);

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

            vec4 out_nx = _mm_mul_ps(_mm_sub_ps(rec.p_x, cur_cen_x), inv_radius);
            vec4 out_ny = _mm_mul_ps(_mm_sub_ps(rec.p_y, cur_cen_y), inv_radius);
            vec4 out_nz = _mm_mul_ps(_mm_sub_ps(rec.p_z, cur_cen_z), inv_radius);

            rec.set_face_normal_4(r_packs, out_nx, out_ny, out_nz);
            rec.mat.fill(mat.get());
            return true;
        }
};

class quad : public hitable {
    public:
        quad(const point3& Q, const vec3& u, const vec3& v, std::shared_ptr<material> mat)
            : Q(Q), u(u), v(v), mat(mat) {        
            vec3 n = cross(u, v);
            float det = dot(n, n);
            vec3 normal = unit_vector(n);
            // Precompute inverse basis rows for alpha and beta barycentric coords:
            // m0 maps to alpha (along u), m1 maps to beta (along v)
            vec3 m0 = cross(v, n) / det;
            vec3 m1 = cross(n, u) / det;

            v_Q_x = _mm_set1_ps(Q.x());
            v_Q_y = _mm_set1_ps(Q.y());
            v_Q_z = _mm_set1_ps(Q.z());

            v_m0_x = _mm_set1_ps(m0.x()); v_m0_y = _mm_set1_ps(m0.y()); v_m0_z = _mm_set1_ps(m0.z());
            v_m1_x = _mm_set1_ps(m1.x()); v_m1_y = _mm_set1_ps(m1.y()); v_m1_z = _mm_set1_ps(m1.z());

            v_normal_x = _mm_set1_ps(normal.x());
            v_normal_y = _mm_set1_ps(normal.y());
            v_normal_z = _mm_set1_ps(normal.z());

            set_bounding_box();
        }

        void set_bounding_box() {
            // Direct vector offset bounding box calculation
            vec3 min_offset(std::min(0.0f, u.x()) + std::min(0.0f, v.x()),std::min(0.0f, u.y()) + std::min(0.0f, v.y()),std::min(0.0f, u.z()) + std::min(0.0f, v.z()));
            vec3 max_offset(std::max(0.0f, u.x()) + std::max(0.0f, v.x()),std::max(0.0f, u.y()) + std::max(0.0f, v.y()),std::max(0.0f, u.z()) + std::max(0.0f, v.z()));
            bbox = aabb4(Q + min_offset, Q + max_offset);
        }

        aabb4 bounding_box() const override { return bbox; }

        bool hit(const Raypackets& r, const Interval4 ray_t, hit_rec& rec) const override {
            // 1. Ray origin relative to Q
            vec4 rel_ox = _mm_sub_ps(r.orig_x, v_Q_x);
            vec4 rel_oy = _mm_sub_ps(r.orig_y, v_Q_y);
            vec4 rel_oz = _mm_sub_ps(r.orig_z, v_Q_z);            
            // 2. Plane depth t = -(normal . (O - Q)) / (normal . D)
            vec4 D_loc_z, O_loc_z;
            dot_simd4(v_normal_x, v_normal_y, v_normal_z, r.dir_x, r.dir_y, r.dir_z, D_loc_z);
            dot_simd4(v_normal_x, v_normal_y, v_normal_z, rel_ox, rel_oy, rel_oz, O_loc_z);
            // Reject rays parallel to plane
            vec4 abs_dz = _mm_andnot_ps(_mm_set1_ps(-0.0f), D_loc_z);
            vec4 valid_denom = _mm_cmpgt_ps(abs_dz, _mm_set1_ps(1e-8f));
            vec4 t = _mm_div_ps(_mm_sub_ps(_mm_setzero_ps(), O_loc_z), D_loc_z);
            vec4 valid_t = _mm_and_ps(_mm_cmpgt_ps(t, ray_t.min), _mm_cmplt_ps(t, ray_t.max));
            vec4 hit_mask = _mm_and_ps(valid_denom, valid_t);
            if (_mm_movemask_ps(hit_mask) == 0) return false;
            // 3. Compute relative hit point directly: rel_p = rel_o + t * D
            vec4 rel_px = _mm_add_ps(rel_ox, _mm_mul_ps(t, r.dir_x));
            vec4 rel_py = _mm_add_ps(rel_oy, _mm_mul_ps(t, r.dir_y));
            vec4 rel_pz = _mm_add_ps(rel_oz, _mm_mul_ps(t, r.dir_z));
            // 4. Project relative hit point to alpha/beta (Only 2 dot products!)
            vec4 alpha, beta;
            dot_simd4(v_m0_x, v_m0_y, v_m0_z, rel_px, rel_py, rel_pz, alpha);
            dot_simd4(v_m1_x, v_m1_y, v_m1_z, rel_px, rel_py, rel_pz, beta);
            // 5. Bounds check: 0 <= alpha <= 1 AND 0 <= beta <= 1
            vec4 zero = _mm_setzero_ps();
            vec4 one  = _mm_set1_ps(1.0f);
            vec4 alpha_valid = _mm_and_ps(_mm_cmpge_ps(alpha, zero), _mm_cmple_ps(alpha, one));
            vec4 beta_valid  = _mm_and_ps(_mm_cmpge_ps(beta, zero),  _mm_cmple_ps(beta, one));
            hit_mask = _mm_and_ps(hit_mask, _mm_and_ps(alpha_valid, beta_valid));
            if (_mm_movemask_ps(hit_mask) == 0) return false;
            // 6. Write hit record
            rec.t = t;
            rec.p_x = _mm_add_ps(v_Q_x, rel_px);
            rec.p_y = _mm_add_ps(v_Q_y, rel_py);
            rec.p_z = _mm_add_ps(v_Q_z, rel_pz);
            rec.hit_mask = hit_mask;
            rec.set_face_normal_4(r, v_normal_x, v_normal_y, v_normal_z);
            // Unconditional branchless pointer stores
            material* raw_mat = mat.get();
            rec.mat[0] = raw_mat;
            rec.mat[1] = raw_mat;
            rec.mat[2] = raw_mat;
            rec.mat[3] = raw_mat;

            return true;
        }

    private:
        point3 Q;
        vec3 u, v;
        std::shared_ptr<material> mat;
        aabb4 bbox;
        // Reduced from 15 to 12 SIMD member registers
        vec4 v_m0_x, v_m0_y, v_m0_z;
        vec4 v_m1_x, v_m1_y, v_m1_z;
        vec4 v_Q_x, v_Q_y, v_Q_z;
        vec4 v_normal_x, v_normal_y, v_normal_z;
};

class triangle : public hitable {
    public:
        triangle(const point3& Q, const vec3& u, const vec3& v, std::shared_ptr<material> mat): Q(Q), u(u), v(v), mat(mat) {        
            vec3 n = cross(u, v);
            float det = dot(n, n);
            vec3 normal = unit_vector(n);
            // Inverse basis rows mapping to barycentric alpha (along u) and beta (along v)
            vec3 m0 = cross(v, n) / det;
            vec3 m1 = cross(n, u) / det;

            v_Q_x = _mm_set1_ps(Q.x());
            v_Q_y = _mm_set1_ps(Q.y());
            v_Q_z = _mm_set1_ps(Q.z());

            v_m0_x = _mm_set1_ps(m0.x()); v_m0_y = _mm_set1_ps(m0.y()); v_m0_z = _mm_set1_ps(m0.z());
            v_m1_x = _mm_set1_ps(m1.x()); v_m1_y = _mm_set1_ps(m1.y()); v_m1_z = _mm_set1_ps(m1.z());

            v_normal_x = _mm_set1_ps(normal.x());
            v_normal_y = _mm_set1_ps(normal.y());
            v_normal_z = _mm_set1_ps(normal.z());

            set_bounding_box();
        }

        void set_bounding_box() {
            // Direct min/max offsets covering origin Q, Q+u, and Q+v
            vec3 min_offset(std::min({0.0f, u.x(), v.x()}),std::min({0.0f, u.y(), v.y()}),std::min({0.0f, u.z(), v.z()}));
            vec3 max_offset(std::max({0.0f, u.x(), v.x()}),std::max({0.0f, u.y(), v.y()}),std::max({0.0f, u.z(), v.z()}));
            bbox = aabb4(Q + min_offset, Q + max_offset);
        }

        aabb4 bounding_box() const override { return bbox; }

        bool hit(const Raypackets& r, const Interval4 ray_t, hit_rec& rec) const override {
            // 1. Ray origin relative to Q
            vec4 rel_ox = _mm_sub_ps(r.orig_x, v_Q_x);
            vec4 rel_oy = _mm_sub_ps(r.orig_y, v_Q_y);
            vec4 rel_oz = _mm_sub_ps(r.orig_z, v_Q_z);            
            // 2. Plane depth t = -(normal . (O - Q)) / (normal . D)
            vec4 D_loc_z, O_loc_z;
            dot_simd4(v_normal_x, v_normal_y, v_normal_z, r.dir_x, r.dir_y, r.dir_z, D_loc_z);
            dot_simd4(v_normal_x, v_normal_y, v_normal_z, rel_ox, rel_oy, rel_oz, O_loc_z);

            vec4 abs_dz = _mm_andnot_ps(_mm_set1_ps(-0.0f), D_loc_z);
            vec4 valid_denom = _mm_cmpgt_ps(abs_dz, _mm_set1_ps(1e-8f));

            vec4 t = _mm_div_ps(_mm_sub_ps(_mm_setzero_ps(), O_loc_z), D_loc_z);
            vec4 valid_t = _mm_and_ps(_mm_cmpgt_ps(t, ray_t.min), _mm_cmplt_ps(t, ray_t.max));
            vec4 hit_mask = _mm_and_ps(valid_denom, valid_t);

            if (_mm_movemask_ps(hit_mask) == 0) return false;

            // 3. Compute relative hit point: rel_p = rel_o + t * D
            vec4 rel_px = _mm_add_ps(rel_ox, _mm_mul_ps(t, r.dir_x));
            vec4 rel_py = _mm_add_ps(rel_oy, _mm_mul_ps(t, r.dir_y));
            vec4 rel_pz = _mm_add_ps(rel_oz, _mm_mul_ps(t, r.dir_z));

            // 4. Project relative hit point to barycentric alpha and beta (2 dot products)
            vec4 alpha, beta;
            dot_simd4(v_m0_x, v_m0_y, v_m0_z, rel_px, rel_py, rel_pz, alpha);
            dot_simd4(v_m1_x, v_m1_y, v_m1_z, rel_px, rel_py, rel_pz, beta);
            // 5. Barycentric triangle check: alpha >= 0, beta >= 0, (alpha + beta) <= 1
            vec4 zero = _mm_setzero_ps();
            vec4 one  = _mm_set1_ps(1.0f);

            vec4 alpha_valid     = _mm_cmpge_ps(alpha, zero);
            vec4 beta_valid      = _mm_cmpge_ps(beta, zero);
            vec4 alpha_plus_beta = _mm_add_ps(alpha, beta);
            vec4 sum_valid       = _mm_cmple_ps(alpha_plus_beta, one);

            hit_mask = _mm_and_ps(hit_mask, _mm_and_ps(alpha_valid, _mm_and_ps(beta_valid, sum_valid)));

            if (_mm_movemask_ps(hit_mask) == 0) return false;
            // 6. Write hit record using rel_p + Q reconstruction
            rec.t = t;
            rec.p_x = _mm_add_ps(v_Q_x, rel_px);
            rec.p_y = _mm_add_ps(v_Q_y, rel_py);
            rec.p_z = _mm_add_ps(v_Q_z, rel_pz);
            rec.hit_mask = hit_mask;
            rec.set_face_normal_4(r, v_normal_x, v_normal_y, v_normal_z);
            // Unconditional branchless pointer stores
            material* raw_mat = mat.get();
            rec.mat[0] = raw_mat;
            rec.mat[1] = raw_mat;
            rec.mat[2] = raw_mat;
            rec.mat[3] = raw_mat;

            return true;
        }

    private:
        point3 Q;
        vec3 u, v;
        std::shared_ptr<material> mat;
        aabb4 bbox;

        vec4 v_m0_x, v_m0_y, v_m0_z;
        vec4 v_m1_x, v_m1_y, v_m1_z;
        vec4 v_Q_x, v_Q_y, v_Q_z;
        vec4 v_normal_x, v_normal_y, v_normal_z;
};

class box3d : public hitable {
    public:
        box3d(const point3& pmin, const point3& pmax, std::shared_ptr<material> mat): pmin(pmin), pmax(pmax), mat(mat) {            
            v_min_x = _mm_set1_ps(pmin.x());
            v_min_y = _mm_set1_ps(pmin.y());
            v_min_z = _mm_set1_ps(pmin.z());
            v_max_x = _mm_set1_ps(pmax.x());
            v_max_y = _mm_set1_ps(pmax.y());
            v_max_z = _mm_set1_ps(pmax.z());
            bbox = aabb4(pmin, pmax);
        }

        aabb4 bounding_box() const override { return bbox; }

        bool hit(const Raypackets& r, const Interval4 ray_t, hit_rec& rec) const override {
            // 1. Compute slab entry/exit t for all 3 axes using precomputed inv_dir
            vec4 tx1 = _mm_mul_ps(_mm_sub_ps(v_min_x, r.orig_x), r.inv_dir_x);
            vec4 tx2 = _mm_mul_ps(_mm_sub_ps(v_max_x, r.orig_x), r.inv_dir_x);
            vec4 txmin = _mm_min_ps(tx1, tx2);
            vec4 txmax = _mm_max_ps(tx1, tx2);

            vec4 ty1 = _mm_mul_ps(_mm_sub_ps(v_min_y, r.orig_y), r.inv_dir_y);
            vec4 ty2 = _mm_mul_ps(_mm_sub_ps(v_max_y, r.orig_y), r.inv_dir_y);
            vec4 tymin = _mm_min_ps(ty1, ty2);
            vec4 tymax = _mm_max_ps(ty1, ty2);

            vec4 tz1 = _mm_mul_ps(_mm_sub_ps(v_min_z, r.orig_z), r.inv_dir_z);
            vec4 tz2 = _mm_mul_ps(_mm_sub_ps(v_max_z, r.orig_z), r.inv_dir_z);
            vec4 tzmin = _mm_min_ps(tz1, tz2);
            vec4 tzmax = _mm_max_ps(tz1, tz2);

            // 2. Compute overall raw entry and clamped interval t across all slabs
            vec4 t_raw_entry = _mm_max_ps(txmin, _mm_max_ps(tymin, tzmin));
            vec4 t_entry     = _mm_max_ps(ray_t.min, t_raw_entry);
            vec4 t_exit      = _mm_min_ps(ray_t.max, _mm_min_ps(txmax, _mm_min_ps(tymax, tzmax)));

            // 3. Valid hit condition: entry <= exit
            vec4 hit_mask = _mm_cmple_ps(t_entry, t_exit);
            int mask_bits = _mm_movemask_ps(hit_mask);
            if (mask_bits == 0) return false;

            // 4. Branchless normal reconstruction
            // Match against t_raw_entry (unclamped) so rays starting inside the box still identify valid face plane normals
            vec4 zero    = _mm_setzero_ps();
            vec4 one     = _mm_set1_ps(1.0f);
            vec4 neg_one = _mm_set1_ps(-1.0f);

            vec4 match_tx1 = _mm_cmpeq_ps(t_raw_entry, tx1);
            vec4 match_tx2 = _mm_cmpeq_ps(t_raw_entry, tx2);
            vec4 match_ty1 = _mm_cmpeq_ps(t_raw_entry, ty1);
            vec4 match_ty2 = _mm_cmpeq_ps(t_raw_entry, ty2);
            vec4 match_tz1 = _mm_cmpeq_ps(t_raw_entry, tz1);
            vec4 match_tz2 = _mm_cmpeq_ps(t_raw_entry, tz2);

            // Derive face normals (-1 or +1 along hit plane axis)
            vec4 norm_x = _mm_blendv_ps(zero, neg_one, match_tx1);
            norm_x      = _mm_blendv_ps(norm_x, one,   match_tx2);
            vec4 norm_y = _mm_blendv_ps(zero, neg_one, match_ty1);
            norm_y      = _mm_blendv_ps(norm_y, one,   match_ty2);
            vec4 norm_z = _mm_blendv_ps(zero, neg_one, match_tz1);
            norm_z      = _mm_blendv_ps(norm_z, one,   match_tz2);

            // 5. Populate hit record
            rec.t = t_entry;
            ray_at_t4(rec.p_x, rec.p_y, rec.p_z, r, t_entry);
            rec.hit_mask = hit_mask;
            rec.set_face_normal_4(r, norm_x, norm_y, norm_z);

            // 6. Unconditional branchless material writes
            material* raw_mat = mat.get();
            rec.mat[0] = raw_mat;
            rec.mat[1] = raw_mat;
            rec.mat[2] = raw_mat;
            rec.mat[3] = raw_mat;

            return true;
        }

    private:
        point3 pmin, pmax;
        std::shared_ptr<material> mat;
        aabb4 bbox;
        vec4 v_min_x, v_min_y, v_min_z;
        vec4 v_max_x, v_max_y, v_max_z;
};