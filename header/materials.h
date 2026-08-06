#pragma once
#include "general.h"
#include "ray.h"
#include "vec3_simd4.h"



class material {
    public:
        virtual ~material() = default;
        virtual vec4 scatter(const Raypackets& r_in, const hit_rec& rec,vec4& atten_r, vec4& atten_g, vec4& atten_b, Raypackets& scatt_pack)const=0;
};

class lambertian_simd4:public material{
    public:
        lambertian_simd4(color &albedo){
            albedo_r = _mm_set1_ps(albedo.x());
            albedo_g = _mm_set1_ps(albedo.y());
            albedo_b = _mm_set1_ps(albedo.z());
        }
        vec4 scatter(const Raypackets& in_pack,const hit_rec& rec,vec4& atten_r, vec4& atten_g, vec4& atten_b,Raypackets &scatt_pack)const override{
            atten_r = albedo_r;
            atten_g = albedo_g;
            atten_b = albedo_b;
            scatt_pack.orig_x = rec.p_x;
            scatt_pack.orig_y = rec.p_y;
            scatt_pack.orig_z = rec.p_z;
            random_unit_vector_simd4(scatt_pack.dir_x, scatt_pack.dir_y, scatt_pack.dir_z);
            scatt_pack.dir_x = _mm_add_ps(scatt_pack.dir_x, rec.nx);
            scatt_pack.dir_y = _mm_add_ps(scatt_pack.dir_y, rec.ny);
            scatt_pack.dir_z = _mm_add_ps(scatt_pack.dir_z,rec.nz);
            scatt_pack.time = in_pack.time;
            scatt_pack.t_max = _mm_set1_ps(infinity);
           return _mm_castsi128_ps(_mm_set1_epi32(-1)); 
        }
    private:
        vec4 albedo_r, albedo_g, albedo_b;    


};