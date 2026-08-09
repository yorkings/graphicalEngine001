#pragma once
#include "vec3_simd4.h"
class Ray {
    private:
        vec3 orig;
        vec3 dir;
        float tm;
    public:        
        Ray(vec3 org, vec3 dir) : Ray(org,dir,0.0f) {}
        Ray(vec3 org, vec3 dir,float tm) : orig(org), dir(dir),tm(tm) {}
        inline point3 origin() const    { return orig; }
        inline vec3 direction() const { return dir; }
        inline float time()const {return tm;}
        point3 at(float t)const{
            return orig + t*dir;
        }
};

struct alignas(16) Raypackets {
    vec4 orig_x, orig_y, orig_z;
    vec4 dir_x,  dir_y,  dir_z;
    vec4 time;
    vec4 t_max;
    Raypackets() = default;
    inline Raypackets(const Ray r[4]) {
        orig_x = _mm_set_ps(r[3].origin().x(), r[2].origin().x(), r[1].origin().x(), r[0].origin().x());
        orig_y = _mm_set_ps(r[3].origin().y(), r[2].origin().y(), r[1].origin().y(), r[0].origin().y());
        orig_z = _mm_set_ps(r[3].origin().z(), r[2].origin().z(), r[1].origin().z(), r[0].origin().z());

        dir_x  = _mm_set_ps(r[3].direction().x(), r[2].direction().x(), r[1].direction().x(), r[0].direction().x());
        dir_y  = _mm_set_ps(r[3].direction().y(), r[2].direction().y(), r[1].direction().y(), r[0].direction().y());
        dir_z  = _mm_set_ps(r[3].direction().z(), r[2].direction().z(), r[1].direction().z(), r[0].direction().z());
        time  = _mm_set_ps(r[3].time(), r[2].time(), r[1].time(), r[0].time());
        t_max  = _mm_set1_ps(1e30f);
    }
};

inline void ray_at_t4(vec4& px, vec4& py, vec4& pz,const Raypackets& ray, vec4 t) {
    px = _mm_add_ps(ray.orig_x, _mm_mul_ps(t, ray.dir_x));
    py = _mm_add_ps(ray.orig_y, _mm_mul_ps(t, ray.dir_y));
    pz = _mm_add_ps(ray.orig_z, _mm_mul_ps(t, ray.dir_z));
}