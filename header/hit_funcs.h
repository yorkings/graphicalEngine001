#pragma once
#include "general.h"
#include "ray.h"
#include "intervals.h"

struct hit_rec{
    __m128 t;             
    __m128 p_x, p_y, p_z; 
    __m128 nx, ny, nz;    
    __m128 hit_mask;
   inline void set_face_normal_4(const Raypackets& r, __m128 out_nx, __m128 out_ny, __m128 out_nz) {
        __m128 dot_x = _mm_mul_ps(r.dir_x, out_nx);
        __m128 dot_y = _mm_mul_ps(r.dir_y, out_ny);
        __m128 dot_z = _mm_mul_ps(r.dir_z, out_nz);
        __m128 dot_prod=_mm_add_ps(_mm_add_ps(dot_x ,dot_y),dot_z);
    
        __m128 is_back_face = _mm_cmpgt_ps(dot_prod, _mm_setzero_ps());
        //Flip the normal sign bit for back-facing rays using XOR (No IF/ELSE branches!)
        __m128 sign_bit = _mm_set1_ps(-0.0f); // Bitmask with only IEEE sign bit set (0x80000000)
        __m128 flip_mask = _mm_and_ps(is_back_face, sign_bit);
        // XORing with 0x80000000 flips the float sign (+ to -, - to +)
        nx = _mm_xor_ps(out_nx, flip_mask);
        ny = _mm_xor_ps(out_ny, flip_mask);
        nz = _mm_xor_ps(out_nz, flip_mask);
    }
};

struct hitable{
    ~hitable()=default;
    virtual bool hit(const Raypackets &r_packs,const Interval4& ray_t,hit_rec& rec)const{return false;}

};

class hit_list{
    public:
        std::vector<shared_ptr<hitable>> objects;

        ~hit_list()=default;
        hit_list(){}
        hit_list(shared_ptr<hitable>object){add(object);}

        void add(shared_ptr<hitable>object){return objects.push_back(object);}
        void clear(){objects.clear();}
        virtual bool hit(const Raypackets& r, Interval4 ray_t, hit_rec& rec) const {
            hit_rec temp_rec;
            bool hit_anything = false;
            __m128 accumulated_hit_mask=_mm_setzero_ps();
            for(const auto &object:objects){
                if(object->hit(r, ray_t, temp_rec)){
                    hit_anything = true;
                    accumulated_hit_mask=_mm_or_ps(accumulated_hit_mask,temp_rec.hit_mask);
                    ray_t.max=blend_ps(ray_t.max, temp_rec.t, temp_rec.hit_mask);
                    update_hit_record(rec, temp_rec, temp_rec.hit_mask);                    
                }
            }
            rec.hit_mask = accumulated_hit_mask;
            return hit_anything;
        }

    private:
        inline void update_hit_record(hit_rec& dst, const hit_rec& src, __m128 lane_mask) const {
            dst.t = blend_ps(dst.t, src.t, lane_mask);            
            dst.p_x = blend_ps(dst.p_x, src.p_x, lane_mask);
            dst.p_y = blend_ps(dst.p_y, src.p_y, lane_mask);
            dst.p_z = blend_ps(dst.p_z, src.p_z, lane_mask);
            dst.nx = blend_ps(dst.nx, src.nx, lane_mask);
            dst.ny = blend_ps(dst.ny, src.ny, lane_mask);
            dst.nz = blend_ps(dst.nz, src.nz, lane_mask);
        }
        // SIMD Bitwise Selection Helper: (mask ? b : a)
        inline __m128 blend_ps(__m128 a, __m128 b, __m128 mask) const {return _mm_blendv_ps(a,b,mask);} 

};  