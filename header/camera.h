#pragma once
#include "general.h"
#include "polygons.h"
#include"color_write.h"
#include "materials.h"

class camera{       
    public:
        std::string filename;
        float aspect_ratio= 16.0f / 9.0f;
        int   image_width =0;
        int image_height=0;
        int   samples_per_pixel  = 10;  
        int   max_depth = 50;        
        // View camera geometry
        float vfov     = 90.0f;           // Vertical field-of-view in degrees
        point3 lookFrom= point3(0, 0, 0); // Point camera is looking from
        point3 lookAt  = point3(0, 0, -1);// Point camera is looking at
        vec3   vup     = vec3(0, 1, 0);   // Camera-relative "up" direction  
        // Depth of Field (Lens controls)
        float defocus_angle= 0.0f;// Variation angle of rays through lens
        float focus_dist= 10.0f;// Distance from lookFrom to plane of perfect focus    
        void render(hit_list& world, std::vector<uint32_t>& screen_pixels); 
    private:
        vec3 camera_center;
        point3 pixel00_loc;        // Location of top-left pixel (Pixel [0,0])
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;//ofset to pixel below
        float pixel_sample_scale;
        int spp_sqrt;
        float recip_sqrt_spp; 
        vec3   u, v, w;            // Camera frame coordinate system
        // Defocus blur (Depth of Field) parameters
        vec3   defocus_disk_u; 
        vec3   defocus_disk_v;
        // Pre-broadcasted SIMD constants 
        vec4 v_cam_center_x, v_cam_center_y, v_cam_center_z;
        vec4 v_p00_x, v_p00_y, v_p00_z;
        vec4 v_pdu_x, v_pdu_y, v_pdu_z;
        vec4 v_pdv_x, v_pdv_y, v_pdv_z;
        vec4 v_def_u_x, v_def_u_y, v_def_u_z;
        vec4 v_def_v_x, v_def_v_y, v_def_v_z;
        vec4 v_recip_sqrt_spp;
        void initialize();       
         
        inline static vec4 sanitize_simd4(const vec4& x){
            const vec4 zero = _mm_setzero_ps();        
            vec4 nan_mask = _mm_cmpunord_ps(x, x);
            vec4 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f),x);        
            vec4 inf_mask = _mm_cmpeq_ps(abs_x,_mm_set1_ps(infinity));        
            vec4 bad_mask = _mm_or_ps(nan_mask, inf_mask);        
            vec4 clean = _mm_blendv_ps(x, zero, bad_mask);        
            return _mm_max_ps(clean, zero);
        }
  
        inline Raypackets get_ray_packs(int i,int j,int s_i = 0, int s_j = 0)const{
           vec4 pix_i=_mm_set_ps(static_cast<float>(i + 1), static_cast<float>(i),static_cast<float>(i + 1), static_cast<float>(i));
           vec4 pix_j=_mm_set_ps(static_cast<float>(j + 1), static_cast<float>(j + 1),static_cast<float>(j), static_cast<float>(j));

            vec4 rand_u, rand_v;
            random_float_simd4(rand_u);
            random_float_simd4(rand_v);

            vec4 s_i_vec    = _mm_set1_ps(static_cast<float>(s_i));
            vec4 s_j_vec    = _mm_set1_ps(static_cast<float>(s_j));
            vec4 offset_x = _mm_mul_ps(_mm_add_ps(s_i_vec, rand_u), v_recip_sqrt_spp);
            vec4 offset_y = _mm_mul_ps(_mm_add_ps(s_j_vec, rand_v), v_recip_sqrt_spp);
            vec4 sample_coord_x = _mm_add_ps(pix_i, offset_x);
            vec4 sample_coord_y = _mm_add_ps(pix_j, offset_y);

            vec4 sample_x = _mm_add_ps(v_p00_x,_mm_add_ps(_mm_mul_ps(sample_coord_x,v_pdu_x),_mm_mul_ps(sample_coord_y,v_pdv_x)));
            vec4 sample_y = _mm_add_ps(v_p00_y, _mm_add_ps(_mm_mul_ps(sample_coord_x, v_pdu_y), _mm_mul_ps(sample_coord_y, v_pdv_y)));
            vec4 sample_z = _mm_add_ps(v_p00_z, _mm_add_ps(_mm_mul_ps(sample_coord_x, v_pdu_z), _mm_mul_ps(sample_coord_y, v_pdv_z)));           

            vec4 orig_x, orig_y, orig_z;
            if(defocus_angle <= 0.0f) {
                orig_x = _mm_set1_ps(camera_center.x());
                orig_y = _mm_set1_ps(camera_center.y());
                orig_z = _mm_set1_ps(camera_center.z());
            } else {
                defocus_disk_sample_simd4(orig_x, orig_y, orig_z);
            } 
            Raypackets ray_packs;
            ray_packs.orig_x=orig_x;
            ray_packs.orig_y=orig_y;
            ray_packs.orig_z=orig_z;
            ray_packs.dir_x=_mm_sub_ps(sample_x,orig_x);
            ray_packs.dir_y=_mm_sub_ps(sample_y,orig_y);
            ray_packs.dir_z=_mm_sub_ps(sample_z,orig_z);  
            
            ray_packs.update_inv_dir();
            random_float_simd4(ray_packs.time); 

            return ray_packs;
        }


        inline void get_background_simd(const Raypackets& pack, vec4& bg_r, vec4& bg_g, vec4& bg_b) const {
            vec4 unit_x, unit_y, unit_z;
            normalize_simd4(pack.dir_x, pack.dir_y, pack.dir_z, unit_x, unit_y, unit_z);
            // t = 0.5 * (unit_y + 1.0)
            vec4 t           = _mm_mul_ps(_mm_set1_ps(0.5f), _mm_add_ps(unit_y, _mm_set1_ps(1.0f)));
            vec4 one_minus_t = _mm_sub_ps(_mm_set1_ps(1.0f), t);
            // color = (1.0 - t)*White + t*SkyBlue(0.5, 0.7, 1.0)
            bg_r = _mm_add_ps(one_minus_t, _mm_mul_ps(t, _mm_set1_ps(0.5f)));
            bg_g = _mm_add_ps(one_minus_t, _mm_mul_ps(t, _mm_set1_ps(0.7f)));
            bg_b = _mm_add_ps(one_minus_t, _mm_mul_ps(t, _mm_set1_ps(1.0f)));
        }
        
        inline vec4 scatter_materials_simd(const Raypackets& in_pack,const hit_rec& rec,const vec4& active_mask,vec4& atten_r, vec4& atten_g, vec4& atten_b,Raypackets& scatt_pack) {
            atten_r = _mm_setzero_ps();
            atten_g = _mm_setzero_ps();
            atten_b = _mm_setzero_ps(); 

            scatt_pack.orig_x = in_pack.orig_x;
            scatt_pack.orig_y = in_pack.orig_y;
            scatt_pack.orig_z = in_pack.orig_z;
            scatt_pack.dir_x  = _mm_setzero_ps();
            scatt_pack.dir_y  = _mm_setzero_ps();
            scatt_pack.dir_z  = _mm_setzero_ps();
            scatt_pack.time   = in_pack.time; 

            vec4 scatter_mask = _mm_setzero_ps();
            int processed_lanes = 0;
            vec4 active_hit_mask =_mm_and_ps(active_mask, rec.hit_mask);
            int hit_bits = _mm_movemask_ps(active_hit_mask);            
            for (int i = 0; i < 4; ++i) {
                if (!(hit_bits & (1 << i)) || (processed_lanes & (1 << i))) continue;                
                const material* mat = rec.mat[i];
                if (!mat) continue;                               
                vec4 match_mask =_mm_castsi128_ps(_mm_set_epi32(
                        rec.mat[3] == mat ? -1 : 0,
                        rec.mat[2] == mat ? -1 : 0,
                        rec.mat[1] == mat ? -1 : 0,
                        rec.mat[0] == mat ? -1 : 0
                    )); 
                match_mask = _mm_and_ps(match_mask,active_hit_mask);                   
                vec4 cur_atten_r, cur_atten_g, cur_atten_b;
                Raypackets cur_scatt;
                vec4 cur_mask = mat->scatter(in_pack, rec, active_hit_mask,cur_atten_r, cur_atten_g, cur_atten_b, cur_scatt);
                cur_mask = _mm_and_ps(cur_mask, match_mask);

                atten_r = _mm_blendv_ps(atten_r, cur_atten_r, cur_mask);
                atten_g = _mm_blendv_ps(atten_g, cur_atten_g, cur_mask);
                atten_b = _mm_blendv_ps(atten_b, cur_atten_b, cur_mask); 

                scatt_pack.orig_x = _mm_blendv_ps(scatt_pack.orig_x, cur_scatt.orig_x, cur_mask);
                scatt_pack.orig_y = _mm_blendv_ps(scatt_pack.orig_y, cur_scatt.orig_y, cur_mask);
                scatt_pack.orig_z = _mm_blendv_ps(scatt_pack.orig_z, cur_scatt.orig_z, cur_mask);                
                scatt_pack.dir_x  = _mm_blendv_ps(scatt_pack.dir_x,  cur_scatt.dir_x,  cur_mask);
                scatt_pack.dir_y  = _mm_blendv_ps(scatt_pack.dir_y,  cur_scatt.dir_y,  cur_mask);
                scatt_pack.dir_z  = _mm_blendv_ps(scatt_pack.dir_z,  cur_scatt.dir_z,  cur_mask);                
                scatt_pack.time   = _mm_blendv_ps(scatt_pack.time,   cur_scatt.time,   cur_mask);  

                scatter_mask = _mm_or_ps(scatter_mask, cur_mask);                
                for (int j = i; j < 4; ++j) {
                    if (rec.mat[j] == mat) processed_lanes |= (1 << j);
                }
            }
            return scatter_mask;
        }

        void ray_color_packet_simd(const Raypackets& pack, const hit_list& world, int depth,vec4& out_r, vec4& out_g, vec4& out_b){
            Raypackets current_packs=pack;
            vec4 tp_r = _mm_set1_ps(1.0f);
            vec4 tp_g = _mm_set1_ps(1.0f);
            vec4 tp_b = _mm_set1_ps(1.0f);

            out_r = _mm_setzero_ps();
            out_g = _mm_setzero_ps();
            out_b = _mm_setzero_ps();

            vec4 active_mask = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
            for (int bounce = 0; bounce < depth; ++bounce) {
                if (_mm_movemask_ps(active_mask) == 0) break;

                hit_rec rec;
                Interval4 ray_t(0.001f, infinity);
                world.hit(current_packs, ray_t, rec);

                vec4 newly_missed = _mm_andnot_ps(rec.hit_mask, active_mask);
                if (_mm_movemask_ps(newly_missed) != 0) {
                    vec4 bg_r, bg_g, bg_b;
                    get_background_simd(current_packs, bg_r, bg_g, bg_b);
                    out_r=_mm_add_ps(out_r,_mm_and_ps(newly_missed, _mm_mul_ps(tp_r, bg_r)));
                    out_g=_mm_add_ps(out_g,_mm_and_ps(newly_missed, _mm_mul_ps(tp_g, bg_g)));
                    out_b=_mm_add_ps(out_b,_mm_and_ps(newly_missed, _mm_mul_ps(tp_b, bg_b)));
                }

                active_mask = _mm_and_ps(active_mask, rec.hit_mask);  
                if (_mm_movemask_ps(active_mask) == 0) break;  
                Raypackets scattered_packs;
                vec4 atten_r, atten_g, atten_b;                
                vec4 scatter_mask = scatter_materials_simd(current_packs, rec,active_mask, atten_r, atten_g, atten_b, scattered_packs); 
                vec4 valid_scatter_mask = _mm_and_ps(active_mask, scatter_mask);

                tp_r = _mm_blendv_ps(tp_r, _mm_mul_ps(tp_r, atten_r), valid_scatter_mask);
                tp_g = _mm_blendv_ps(tp_g, _mm_mul_ps(tp_g, atten_g), valid_scatter_mask);
                tp_b = _mm_blendv_ps(tp_b, _mm_mul_ps(tp_b, atten_b), valid_scatter_mask);

                current_packs.orig_x = _mm_blendv_ps(current_packs.orig_x, scattered_packs.orig_x, valid_scatter_mask);
                current_packs.orig_y = _mm_blendv_ps(current_packs.orig_y, scattered_packs.orig_y, valid_scatter_mask);
                current_packs.orig_z = _mm_blendv_ps(current_packs.orig_z, scattered_packs.orig_z, valid_scatter_mask);
                current_packs.dir_x  = _mm_blendv_ps(current_packs.dir_x,  scattered_packs.dir_x,  valid_scatter_mask);
                current_packs.dir_y  = _mm_blendv_ps(current_packs.dir_y,  scattered_packs.dir_y,  valid_scatter_mask);
                current_packs.dir_z  = _mm_blendv_ps(current_packs.dir_z,  scattered_packs.dir_z,  valid_scatter_mask);
                current_packs.time   = _mm_blendv_ps(current_packs.time,   scattered_packs.time,   valid_scatter_mask);
                current_packs.update_inv_dir();

                active_mask = _mm_and_ps(active_mask, scatter_mask);                
            }

        }

        //Shirley's Concentric Disk Mapping
       inline void defocus_disk_sample_simd4(__m128& out_orig_x, __m128& out_orig_y, __m128& out_orig_z) const {
            __m128 u1, u2;
            random_float_simd4(u1); 
            random_float_simd4(u2);

            // Radius r = sqrt(u1), Angle theta = 2 * pi * u2
            __m128 r = _mm_sqrt_ps(u1);
            __m128 theta = _mm_mul_ps(_mm_set1_ps(2.0f * pi), u2);
            __m128 sin_th, cos_th;
            sincos_simd4(theta, sin_th, cos_th);
            __m128 res_x = _mm_mul_ps(r, cos_th); 
            __m128 res_y = _mm_mul_ps(r, sin_th);
            // Transform disk sample into 3D world-space lens offset
            out_orig_x = _mm_add_ps(v_cam_center_x, _mm_add_ps(_mm_mul_ps(res_x, v_def_u_x), _mm_mul_ps(res_y, v_def_v_x)));
            out_orig_y = _mm_add_ps(v_cam_center_y, _mm_add_ps(_mm_mul_ps(res_x, v_def_u_y), _mm_mul_ps(res_y, v_def_v_y)));
            out_orig_z = _mm_add_ps(v_cam_center_z, _mm_add_ps(_mm_mul_ps(res_x, v_def_u_z), _mm_mul_ps(res_y, v_def_v_z)));
        }
};
