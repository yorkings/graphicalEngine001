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
        
        void render(hit_list& world, std::vector<uint32_t>& screen_pixels){
            initialize();
            if (screen_pixels.size() != static_cast<size_t>(image_width * image_height)) {
                screen_pixels.resize(image_width * image_height);
            }
            std::mdspan pixels_2d(screen_pixels.data(), image_height, image_width);
            int num_threads = 0;
            #pragma omp parallel
            {
                #pragma omp single
                num_threads = omp_get_num_threads();
            }
            std::cerr << "\n=== Rendering Configuration ===" << std::endl;
            std::cerr << "Image: " << image_width << "x" << image_height << std::endl;
            std::cerr << "Samples: " << samples_per_pixel << " (" << spp_sqrt << "x" << spp_sqrt << " stratified)" << std::endl;
            std::cerr << "Threads: " << num_threads << " (SIMD RayPackets Enabled)" << std::endl;
            std::cerr << "================================" << std::endl;
            std::cerr << "\nProgress:" << std::endl;
            std::atomic<int> rows_completed(0);
            int total_rows = image_height;

            #pragma omp parallel for schedule(dynamic, 1)
            for (int j = 0; j < image_height; j += 2) {
                int thread_id = omp_get_thread_num();
                int num_threads_local = omp_get_num_threads();
                for (int i = 0; i < image_width; i += 2) {
                    __m128 accum_r = _mm_setzero_ps();
                    __m128 accum_g = _mm_setzero_ps();
                    __m128 accum_b = _mm_setzero_ps();                
                    // Stratified subpixel sampling
                    for (int s_j = 0; s_j < spp_sqrt; ++s_j) {
                        for (int s_i = 0; s_i < spp_sqrt; ++s_i) {
                            Raypackets pack = get_ray_packs(i, j, s_i, s_j);                        
                            __m128 pack_r, pack_g, pack_b;
                            ray_color_packet_simd(pack, world, max_depth, pack_r, pack_g, pack_b);
                            accum_r = _mm_add_ps(accum_r, pack_r);
                            accum_g = _mm_add_ps(accum_g, pack_g);
                            accum_b = _mm_add_ps(accum_b, pack_b);

                        }
                    }
                    // Vectorized color normalization, gamma correction, and 32-bit packing
                    uint32_t packed_pixels[4];
                    write_color_simd4(accum_r, accum_g, accum_b, pixel_sample_scale, packed_pixels);
                    pixels_2d[j,i]=packed_pixels[0];
                    if (i + 1 < image_width) {
                        pixels_2d[j, i + 1] = packed_pixels[1];
                    }
                    if (j + 1 < image_height) {
                        pixels_2d[j + 1, i] = packed_pixels[2];
                    }
                    if (i + 1 < image_width && j + 1 < image_height) {
                        pixels_2d[j + 1, i + 1] = packed_pixels[3];
                    }
                }
                // Update progress (atomic to avoid race conditions)
                int completed = rows_completed.fetch_add(2) + 2;
                completed = std::min(completed, total_rows);
                if (completed % std::max(1, total_rows / 50) == 0 || completed == total_rows) {
                    int percent = (completed * 100) / total_rows;
                    int bar_width = 40;
                    int filled = (percent * bar_width) / 100;                    
                    #pragma omp critical
                    {
                        std::cerr << "\r[";
                        for (int k = 0; k < bar_width; ++k) {
                            if (k < filled) std::cerr << "=";
                            else if (k == filled) std::cerr << ">";
                            else std::cerr << " ";
                        }
                        std::cerr << "] " << percent << "%  "<< "Rows: " << completed << "/" << total_rows<< std::flush;
                    }
                }                    
            }
            std::cerr << "\n\nRendering complete.\n";
        }
    private:
        vec3 camera_center;
        point3 pixel00_loc;        // Location of top-left pixel (Pixel [0,0])
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;//ofset to pixel below
        float pixel_sample_scale;
        int spp_sqrt;
        float recip_sqrt_spp; 
        //system camera functions
        vec3   u, v, w;            // Camera frame coordinate system
        // Defocus blur (Depth of Field) parameters
        vec3   defocus_disk_u; 
        vec3   defocus_disk_v;

        void initialize(){
            if(image_width>0 && image_height<=0){
                image_height=static_cast<int>(image_width/aspect_ratio);
            }
            spp_sqrt = static_cast<int>(std::sqrt(samples_per_pixel));
            recip_sqrt_spp = 1.0f / static_cast<float>(spp_sqrt);
            pixel_sample_scale = 1.0f / static_cast<float>(samples_per_pixel);
            camera_center=lookFrom;
            //viewport dimensions
            float theta       = degrees_to_radians(vfov);
            float h           = std::tan(theta / 2.0f);
            float viewport_h  = 2.0f * h;
            float viewport_w  = viewport_h * aspect_ratio;
            //camera coordinates
            w=unit_vector(lookFrom-lookAt);
            u=unit_vector(cross(vup,w));
            v=cross(w, u);
            auto viewport_u=viewport_w*u;
            auto viewport_v=viewport_h*-v;//Downward on screen Y-axis
            pixel_delta_u = viewport_u / static_cast<float>(image_width);
            pixel_delta_v = viewport_v / static_cast<float>(image_height);
            // Calculate upper-left pixel center (Pixel 0,0)
            vec3 viewport_upper_left = camera_center - (w) - (viewport_u * 0.5f) - (viewport_v * 0.5f);
            pixel00_loc= viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
            auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2.0f));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;            

        }
        inline Ray get_ray(int i, int j, int s_i = 0, int s_j = 0) const {
            float offset_x = (s_i + random_float()) * recip_sqrt_spp;
            float offset_y = (s_j + random_float()) * recip_sqrt_spp;

            auto pixel_sample = pixel00_loc + ((i + offset_x) * pixel_delta_u) + ((j + offset_y) * pixel_delta_v);
            auto ray_origin = (defocus_angle <= 0) ? camera_center :defocus_disk_sample();
            auto ray_direction = pixel_sample - ray_origin;
            auto ray_time = random_float();

            return Ray(ray_origin, ray_direction, ray_time);
        }
        inline Raypackets get_ray_packs(int i,int j,int s_i = 0, int s_j = 0)const{
            Ray r[4]={get_ray(i,j,s_i,s_j),
                    get_ray(i+1,j,s_i,s_j),
                    get_ray(i,j+1,s_i,s_j),
                    get_ray(i+1,j+1,s_i,s_j)
                };
            return Raypackets(r);
        }
        inline void get_background_simd(const Raypackets& pack, __m128& bg_r, __m128& bg_g, __m128& bg_b) const {
            __m128 dir_lensq   = _mm_add_ps(_mm_mul_ps(pack.dir_x, pack.dir_x),_mm_add_ps(_mm_mul_ps(pack.dir_y, pack.dir_y),_mm_mul_ps(pack.dir_z, pack.dir_z)));
            __m128 inv_dir_len = _mm_rsqrt_ps(dir_lensq);
            __m128 unit_y      = _mm_mul_ps(pack.dir_y, inv_dir_len);            
            // t = 0.5 * (unit_y + 1.0)
            __m128 t           = _mm_mul_ps(_mm_set1_ps(0.5f), _mm_add_ps(unit_y, _mm_set1_ps(1.0f)));
            __m128 one_minus_t = _mm_sub_ps(_mm_set1_ps(1.0f), t);
            // color = (1.0 - t)*White + t*SkyBlue(0.5, 0.7, 1.0)
            bg_r = _mm_add_ps(_mm_mul_ps(one_minus_t, _mm_set1_ps(1.0f)), _mm_mul_ps(t, _mm_set1_ps(0.5f)));
            bg_g = _mm_add_ps(_mm_mul_ps(one_minus_t, _mm_set1_ps(1.0f)), _mm_mul_ps(t, _mm_set1_ps(0.7f)));
            bg_b = _mm_add_ps(_mm_mul_ps(one_minus_t, _mm_set1_ps(1.0f)), _mm_mul_ps(t, _mm_set1_ps(1.0f)));
        }
        void ray_color_packet_simd(const Raypackets& pack, const hit_list& world, int depth,__m128& out_r, __m128& out_g, __m128& out_b){
            Raypackets current_packs=pack;
            //making all have white color
            __m128 tp_r = _mm_set1_ps(1.0f);
            __m128 tp_g = _mm_set1_ps(1.0f);
            __m128 tp_b = _mm_set1_ps(1.0f);
            // Total Accumulated Light initialized to 0.0 (Black)
            out_r = _mm_setzero_ps();
            out_g = _mm_setzero_ps();
            out_b = _mm_setzero_ps();
            // Active mask for 4 parallel ray lanes (0xFFFFFFFF per lane initially)
            __m128 active_mask = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
            for (int bounce = 0; bounce < depth; ++bounce) {
                // Stop early if all 4 rays in the packet have terminated/missed
                if (_mm_movemask_ps(active_mask) == 0) break;
                hit_rec rec;
                Interval4 ray_t(0.001f, infinity);
                world.hit(current_packs, ray_t, rec);
                // Identify active rays that missed geometry on this bounce
                __m128 newly_missed = _mm_andnot_ps(rec.hit_mask, active_mask);
                if (_mm_movemask_ps(newly_missed) != 0) {
                    __m128 bg_r, bg_g, bg_b;
                    get_background_simd(current_packs, bg_r, bg_g, bg_b);
                    // Add (Throughput * Background) ONLY to newly missed lanes
                    out_r=_mm_add_ps(out_r,_mm_and_ps(newly_missed, _mm_mul_ps(tp_r, bg_r)));
                    out_g=_mm_add_ps(out_g,_mm_and_ps(newly_missed, _mm_mul_ps(tp_g, bg_g)));
                    out_b=_mm_add_ps(out_b,_mm_and_ps(newly_missed, _mm_mul_ps(tp_b, bg_b)));
                }
                // Turn off lanes that missed geometry
                active_mask = _mm_and_ps(active_mask, rec.hit_mask);  
                if (_mm_movemask_ps(active_mask) == 0) break;  
                //material scattering &atenuation
                //Raypackets scattered_packs;
                //__m128 atten_r, atten_g, atten_b;
                //__m128 scatter_mask; 

                //scatter_mask = world.scatter(current_packs, rec, atten_r, atten_g, atten_b, scattered_packs);   
                //  Update Active Throughput for Hit Lanes
                //tp_r = _mm_mul_ps(tp_r, atten_r);
                //tp_g = _mm_mul_ps(tp_g, atten_g);
                //tp_b = _mm_mul_ps(tp_b, atten_b);

                // Set Up Rays for Next Bounce
                //current_packs = scattered_packs;
                // Deactivate lanes where rays were absorbed (e.g., black materials or terminated scattering)
                //active_mask = _mm_and_ps(active_mask, scatter_mask);   
                
                // For demonstration, we will terminate rays after the first bounce and accumulate the color from the hit point
                // 1. Multiply throughput by 0.1 (or 0.5 for 50% reflectance)
                __m128 atten = _mm_set1_ps(0.1f); 
                tp_r = _mm_mul_ps(tp_r, atten);
                tp_g = _mm_mul_ps(tp_g, atten);
                tp_b = _mm_mul_ps(tp_b, atten);

                // 2. Set new ray origins to hit points (rec.p)
                current_packs.orig_x = rec.p_x;
                current_packs.orig_y = rec.p_y;
                current_packs.orig_z = rec.p_z;

                // 3. Generate random unit vectors for 4 lanes
                vec3 rand0 = random_unit_vector();
                vec3 rand1 = random_unit_vector();
                vec3 rand2 = random_unit_vector();
                vec3 rand3 = random_unit_vector();

                __m128 rand_x = _mm_set_ps(rand3.x(), rand2.x(), rand1.x(), rand0.x());
                __m128 rand_y = _mm_set_ps(rand3.y(), rand2.y(), rand1.y(), rand0.y());
                __m128 rand_z = _mm_set_ps(rand3.z(), rand2.z(), rand1.z(), rand0.z());

                // 4. Set new direction = normal + random_unit_vector()
                current_packs.dir_x = _mm_add_ps(rec.nx, rand_x);
                current_packs.dir_y = _mm_add_ps(rec.ny, rand_y);
                current_packs.dir_z = _mm_add_ps(rec.nz, rand_z);
            }

        }

        point3 defocus_disk_sample() const {
            auto p = random_in_unit_disk();
            return camera_center + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
        }     
};