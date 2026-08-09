#pragma once
#include "general.h"
#include "ray.h"
#include "hit_funcs.h"

struct material {
        virtual ~material() = default;
        virtual vec4 scatter(const Raypackets& r_in, const hit_rec& rec,vec4& atten_r, vec4& atten_g, vec4& atten_b, Raypackets& scatt_pack)const=0;
};

class lambertian_simd4:public material{
    public:
        lambertian_simd4(const color &albedo){
            albedo_r = _mm_set1_ps(albedo.x());
            albedo_g = _mm_set1_ps(albedo.y());
            albedo_b = _mm_set1_ps(albedo.z());
        }
        vec4 scatter(const Raypackets& in_pack, const hit_rec &rec, vec4& atten_r, vec4& atten_g, vec4& atten_b, Raypackets &scatt_pack) const override {
            atten_r = albedo_r;
            atten_g = albedo_g;
            atten_b = albedo_b;

            scatt_pack.orig_x = rec.p_x;
            scatt_pack.orig_y = rec.p_y;
            scatt_pack.orig_z = rec.p_z;

            vec4 rand_x, rand_y, rand_z;
            random_unit_vector_simd4(rand_x, rand_y, rand_z);

            vec4 dir_x = _mm_add_ps(rand_x, rec.nx);
            vec4 dir_y = _mm_add_ps(rand_y, rec.ny);
            vec4 dir_z = _mm_add_ps(rand_z, rec.nz);
            vec4 dir_sq; 
            dot_simd4(dir_x, dir_y, dir_z, dir_x, dir_y, dir_z,dir_sq);
            vec4 is_zero = _mm_cmplt_ps(dir_sq, _mm_set1_ps(1e-8f));

            // Fall back to surface normal if direction is degenerate
            scatt_pack.dir_x = _mm_blendv_ps(dir_x, rec.nx, is_zero);
            scatt_pack.dir_y = _mm_blendv_ps(dir_y, rec.ny, is_zero);
            scatt_pack.dir_z = _mm_blendv_ps(dir_z, rec.nz, is_zero);

            scatt_pack.time = in_pack.time;
            scatt_pack.t_max = _mm_set1_ps(infinity);

            return _mm_castsi128_ps(_mm_set1_epi32(-1)); // Always scatter
        }
    private:
        vec4 albedo_r, albedo_g, albedo_b;    

};

class metal_simd4:public material{
    public:
        metal_simd4(const color &albedo,float fuzzy){
            albedo_r = _mm_set1_ps(albedo.x());
            albedo_g = _mm_set1_ps(albedo.y());
            albedo_b = _mm_set1_ps(albedo.z());
            fuzz=_mm_set1_ps(fuzzy< 1.0f ? fuzzy : 1.0f);
        }
        vec4 scatter(const Raypackets& in_pack,const hit_rec& rec,vec4& atten_r, vec4& atten_g, vec4& atten_b,Raypackets &scatt_pack)const override{
            vec4 unit_x,unit_y,unit_z;
            normalize_simd4(in_pack.dir_x,in_pack.dir_y,in_pack.dir_z,unit_x,unit_y,unit_z);
            vec4 refl_x, refl_y, refl_z;
            reflect_simd4(unit_x, unit_y, unit_z, rec.nx, rec.ny, rec.nz, refl_x, refl_y, refl_z);
            vec4 rand_x, rand_y, rand_z;
            random_unit_vector_simd4(rand_x, rand_y, rand_z);
            scatt_pack.dir_x = _mm_add_ps(refl_x, _mm_mul_ps(fuzz, rand_x));
            scatt_pack.dir_y = _mm_add_ps(refl_y, _mm_mul_ps(fuzz, rand_y));
            scatt_pack.dir_z = _mm_add_ps(refl_z, _mm_mul_ps(fuzz, rand_z));

            scatt_pack.orig_x = rec.p_x;
            scatt_pack.orig_y = rec.p_y;
            scatt_pack.orig_z = rec.p_z;
            scatt_pack.time = in_pack.time;
            scatt_pack.t_max = _mm_set1_ps(infinity);

            atten_r = albedo_r;
            atten_g = albedo_g;
            atten_b = albedo_b;
            vec4 scatt_dot_n;
            dot_simd4(scatt_pack.dir_x, scatt_pack.dir_y, scatt_pack.dir_z, rec.nx, rec.ny, rec.nz, scatt_dot_n);
            return _mm_cmpgt_ps(scatt_dot_n, _mm_setzero_ps());  

        }
    private:
        vec4 albedo_r, albedo_g, albedo_b;
        vec4 fuzz;    

};

class dielectric_simd4 : public material {
    public:
        dielectric_simd4(float index_of_refraction):ior(_mm_set1_ps(index_of_refraction)),inv_ior(_mm_set1_ps(1.0f / index_of_refraction)) {}
        vec4 scatter(const Raypackets& in_pack, const hit_rec& rec,vec4& atten_r, vec4& atten_g, vec4& atten_b,Raypackets &scatt_pack) const override {
            atten_r = _mm_set1_ps(1.0f);
            atten_g = _mm_set1_ps(1.0f);
            atten_b = _mm_set1_ps(1.0f);
            vec4 ri = _mm_blendv_ps(ior,inv_ior, rec.front_face);
            vec4 unit_x, unit_y, unit_z;
            normalize_simd4(in_pack.dir_x, in_pack.dir_y, in_pack.dir_z, unit_x, unit_y, unit_z);
            vec4 unit_dot_n;
            dot_simd4(unit_x, unit_y, unit_z, rec.nx, rec.ny, rec.nz, unit_dot_n);

            vec4 neg_dot = _mm_sub_ps(_mm_setzero_ps(), unit_dot_n);
            vec4 cos_theta = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(neg_dot, _mm_set1_ps(1.0f)));

            vec4 sin_sq = _mm_max_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(cos_theta, cos_theta)), _mm_setzero_ps());
            vec4 sin_theta = _mm_sqrt_ps(sin_sq);

            // Total Internal Reflection condition (ri * sin_theta > 1.0)
            vec4 cannot_refract = _mm_cmpgt_ps(_mm_mul_ps(ri, sin_theta), _mm_set1_ps(1.0f));
            vec4 refl_prob = reflectance_simd4(cos_theta, ri);
            vec4 rand_val;
            random_float_simd4(rand_val);
            vec4 reflect_mask = _mm_or_ps(cannot_refract, _mm_cmpgt_ps(refl_prob, rand_val));
            // Compute candidates for reflection and refraction
            vec4 refl_x, refl_y, refl_z;
            reflect_simd4(unit_x, unit_y, unit_z, rec.nx, rec.ny, rec.nz, refl_x, refl_y, refl_z);            
            vec4 refr_x, refr_y, refr_z;
            refract_simd4(unit_x, unit_y, unit_z, rec.nx, rec.ny, rec.nz, cos_theta, ri, refr_x, refr_y, refr_z);

            scatt_pack.dir_x = _mm_blendv_ps(refr_x, refl_x, reflect_mask);
            scatt_pack.dir_y = _mm_blendv_ps(refr_y, refl_y, reflect_mask);
            scatt_pack.dir_z = _mm_blendv_ps(refr_z, refl_z, reflect_mask);

            scatt_pack.orig_x = rec.p_x;
            scatt_pack.orig_y = rec.p_y;
            scatt_pack.orig_z = rec.p_z;
            scatt_pack.time = in_pack.time;
            scatt_pack.t_max = _mm_set1_ps(infinity);

            return _mm_castsi128_ps(_mm_set1_epi32(-1));
        }
    private:
        vec4 ior; // Index of Refraction   
        vec4 inv_ior;
        static inline vec4 reflectance_simd4(const vec4& cosine, const vec4& refraction_index){
            vec4 safe_cos = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(cosine, _mm_set1_ps(1.0f)));
            //Schlick's approximation: R(theta) = r0 + (1 - r0)*(1 - cos_theta)^5
            vec4 r0=_mm_div_ps(_mm_sub_ps(_mm_set1_ps(1.0f), refraction_index), _mm_add_ps(_mm_set1_ps(1.0f), refraction_index));
            r0 = _mm_mul_ps(r0, r0);
            vec4 omc = _mm_sub_ps(_mm_set1_ps(1.0f), safe_cos);
            vec4 omc2=_mm_mul_ps(omc, omc);
            vec4 omc5 = _mm_mul_ps(_mm_mul_ps(omc2, omc2), omc);
            return _mm_add_ps(r0, _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), r0), omc5));
        } 
};