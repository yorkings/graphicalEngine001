#include "../header/camera.h"

void camera::render(hit_list& world, std::vector<uint32_t>& screen_pixels){
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

    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < image_height; j += 2) {
        int thread_id = omp_get_thread_num();
        int num_threads_local = omp_get_num_threads();
        for (int i = 0; i < image_width; i += 2) {
            seed_rng_simd4_quad(i, j,0);
            vec4 accum_r = _mm_setzero_ps();
            vec4 accum_g = _mm_setzero_ps();
            vec4 accum_b = _mm_setzero_ps();              
            // Stratified subpixel sampling                    
            for (int s_j = 0; s_j < spp_sqrt; ++s_j) {
                for (int s_i = 0; s_i < spp_sqrt; ++s_i) {
                    Raypackets pack = get_ray_packs(i, j, s_i, s_j);                        
                    vec4 pack_r, pack_g, pack_b;
                    ray_color_packet_simd(pack, world, max_depth, pack_r, pack_g, pack_b);
                    accum_r = _mm_add_ps(accum_r, pack_r);
                    accum_g = _mm_add_ps(accum_g, pack_g);
                    accum_b = _mm_add_ps(accum_b, pack_b);
                }
            }
            accum_r = sanitize_simd4(accum_r);
            accum_g = sanitize_simd4(accum_g);
            accum_b = sanitize_simd4(accum_b);
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
                std::cerr << "] " << percent << "%  "<< "Rows: " << completed << "/" << total_rows <<"  Thread: " << thread_id << "/" << num_threads_local<< std::flush;
            }
        }                    
    }
    std::cerr << "\n\nRendering complete.\n";
}

void camera::initialize(){
    if(image_width>0 && image_height<=0){
        image_height=static_cast<int>(image_width/aspect_ratio);
    }else if(image_width<=0 &&image_height>0){
        image_width=static_cast<int>(image_height*aspect_ratio);
    }
    spp_sqrt = static_cast<int>(std::sqrt(samples_per_pixel));
    recip_sqrt_spp = 1.0f / static_cast<float>(spp_sqrt);
    pixel_sample_scale = 1.0f /(spp_sqrt * spp_sqrt);
    camera_center=lookFrom;
    //viewport dimensions
    float theta       = degrees_to_radians(vfov);
    float h           = std::tan(theta / 2.0f);
    float viewport_h  = 2.0f * h*focus_dist;
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
    vec3 viewport_upper_left = camera_center - (focus_dist *w) - (viewport_u * 0.5f) - (viewport_v * 0.5f);
    pixel00_loc= viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);           
    auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2.0f));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
    // Broadcast camera vectors to SIMD registers once during setup
    v_cam_center_x   = _mm_set1_ps(camera_center.x());
    v_cam_center_y   = _mm_set1_ps(camera_center.y());
    v_cam_center_z   = _mm_set1_ps(camera_center.z());

    v_p00_x          = _mm_set1_ps(pixel00_loc.x());
    v_p00_y          = _mm_set1_ps(pixel00_loc.y());
    v_p00_z          = _mm_set1_ps(pixel00_loc.z());

    v_pdu_x          = _mm_set1_ps(pixel_delta_u.x());
    v_pdu_y          = _mm_set1_ps(pixel_delta_u.y());
    v_pdu_z          = _mm_set1_ps(pixel_delta_u.z());

    v_pdv_x          = _mm_set1_ps(pixel_delta_v.x());
    v_pdv_y          = _mm_set1_ps(pixel_delta_v.y());
    v_pdv_z          = _mm_set1_ps(pixel_delta_v.z());

    v_def_u_x        = _mm_set1_ps(defocus_disk_u.x());
    v_def_u_y        = _mm_set1_ps(defocus_disk_u.y());
    v_def_u_z        = _mm_set1_ps(defocus_disk_u.z());

    v_def_v_x        = _mm_set1_ps(defocus_disk_v.x());
    v_def_v_y        = _mm_set1_ps(defocus_disk_v.y());
    v_def_v_z        = _mm_set1_ps(defocus_disk_v.z());

    v_recip_sqrt_spp = _mm_set1_ps(recip_sqrt_spp);            

}