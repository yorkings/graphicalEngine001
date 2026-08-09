#pragma once
#include "general.h"
#include "ray.h"
#include "intervals.h"

class material;

struct hit_rec{
    vec4 t;             
    vec4 p_x, p_y, p_z; 
    vec4 nx, ny, nz;    
    vec4 hit_mask;
    vec4 front_face;
    std::array<const material*,4> mat;
    hit_rec():t(_mm_set1_ps(infinity)),p_x(_mm_setzero_ps()),p_y(_mm_setzero_ps()),p_z(_mm_setzero_ps()),
        nx(_mm_setzero_ps()),ny(_mm_setzero_ps()),nz(_mm_setzero_ps()),hit_mask(_mm_setzero_ps()),front_face(_mm_setzero_ps()){
            mat.fill(nullptr);
        }
   inline void set_face_normal_4(const Raypackets& r, vec4 out_nx, vec4 out_ny, vec4 out_nz) {
        vec4 dot_prod;
        dot_simd4(r.dir_x,r.dir_y,r.dir_z,out_nx,out_ny,out_nz,dot_prod);     
        front_face = _mm_cmplt_ps(dot_prod, _mm_setzero_ps());
        vec4 is_back_face =_mm_xor_ps(front_face, _mm_castsi128_ps(_mm_set1_epi32(-1))); // Invert front_face to get back_face mask
        //Flip the normal sign bit for back-facing rays using XOR 
        vec4 sign_bit = _mm_set1_ps(-0.0f); 
        vec4 flip_mask = _mm_and_ps(is_back_face, sign_bit);
        // XORing with 0x80000000 flips the float sign (+ to -, - to +)
        nx = _mm_xor_ps(out_nx, flip_mask);
        ny = _mm_xor_ps(out_ny, flip_mask);
        nz = _mm_xor_ps(out_nz, flip_mask);
    }
};

struct hitable{
    virtual ~hitable()=default;
    virtual bool hit(const Raypackets &r_packs,const Interval4& ray_t,hit_rec& rec)const{return false;}

};

class hit_list:public hitable{
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
            vec4 accumulated_hit_mask=_mm_setzero_ps();
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
        inline void update_hit_record(hit_rec& dst, const hit_rec& src, vec4 lane_mask) const {
            dst.t = blend_ps(dst.t, src.t, lane_mask);            
            dst.p_x = blend_ps(dst.p_x, src.p_x, lane_mask);
            dst.p_y = blend_ps(dst.p_y, src.p_y, lane_mask);
            dst.p_z = blend_ps(dst.p_z, src.p_z, lane_mask);
            dst.nx = blend_ps(dst.nx, src.nx, lane_mask);
            dst.ny = blend_ps(dst.ny, src.ny, lane_mask);
            dst.nz = blend_ps(dst.nz, src.nz, lane_mask);
            dst.front_face = blend_ps(dst.front_face, src.front_face, lane_mask);

            int mask_bits = _mm_movemask_ps(lane_mask);    
            if (mask_bits & 1) dst.mat[0] = src.mat[0];
            if (mask_bits & 2) dst.mat[1] = src.mat[1];
            if (mask_bits & 4) dst.mat[2] = src.mat[2];
            if (mask_bits & 8) dst.mat[3] = src.mat[3];
        }
        inline vec4 blend_ps(vec4 a, vec4 b, vec4 mask) const {return _mm_blendv_ps(a,b,mask);} 

};  