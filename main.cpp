#include "header/general.h"
#include "header/polygons.h"
#include "header/camera.h"
#include "header/bvh_node.h"
#include "header/utils/helpers.h"


int main() {
    RenderQueries user_config = fetch_user_queries();

    vkb::Instance vkb_instance;
    if (!init_vulkan_compute(vkb_instance)) {
        return -1;
    }
    // Setup Scene & Camera
    hit_list world;
    auto ground_material = make_shared<lambertian_simd4>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for(int a = -11; a < 11; a++){
        for (int b = -11; b < 11; b += 4) {
            vec4 vec_choose_mat, vec_off_a, vec_off_b;
            random_float_simd4(vec_choose_mat);
            random_float_simd4(vec_off_a);
            random_float_simd4(vec_off_b);
            alignas(16) float choose_mat[4];
            alignas(16) float off_a[4];
            alignas(16) float off_b[4];
            _mm_store_ps(choose_mat, vec_choose_mat);
            _mm_store_ps(off_a, vec_off_a);
            _mm_store_ps(off_b, vec_off_b);
            int lanes = std::min(4, 11 - b);
            for (int k = 0; k < lanes; ++k) {
                float cur_b = static_cast<float>(b + k);
                point3 center(a + 0.9f * off_a[k], 0.2f, cur_b + 0.9f * off_b[k]);
                if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                    shared_ptr<material> sphere_mat;
                    if (choose_mat[k] < 0.8f) {
                        //difuse material
                        vec4 r1, g1, b1, r2, g2, b2,radn;
                        random_float_simd4(r1); random_float_simd4(g1); random_float_simd4(b1);
                        random_float_simd4(r2); random_float_simd4(g2); random_float_simd4(b2);
                        random_float_simd4(radn,0.0f,0.5f);

                        alignas(16) float ar1[4], ag1[4], ab1[4], ar2[4], ag2[4], ab2[4],randno[4];
                        _mm_store_ps(ar1, r1); _mm_store_ps(ag1, g1); _mm_store_ps(ab1, b1);
                        _mm_store_ps(ar2, r2); _mm_store_ps(ag2, g2); _mm_store_ps(ab2, b2);
                        _mm_store_ps(randno,radn);

                        color albedo(ar1[k] * ar2[k], ag1[k] * ag2[k], ab1[k] * ab2[k]);
                        sphere_mat = make_shared<lambertian_simd4>(albedo);
                        auto center_2=center +vec3(0,randno[k],0);  
                        world.add(make_shared<sphere>(center,center_2, 0.2f, sphere_mat));                      
                    }
                    else if (choose_mat[k] < 0.95f) {
                        // Metal material
                        vec4 r, g, b_col, fuzz_vec;
                        random_float_simd4(r, 0.5f, 1.0f);
                        random_float_simd4(g, 0.5f, 1.0f);
                        random_float_simd4(b_col, 0.5f, 1.0f);
                        random_float_simd4(fuzz_vec, 0.0f, 0.5f);

                        alignas(16) float ar[4], ag[4], ab[4], afuzz[4];
                        _mm_store_ps(ar, r); _mm_store_ps(ag, g); 
                        _mm_store_ps(ab, b_col); _mm_store_ps(afuzz, fuzz_vec);

                        color albedo(ar[k], ag[k], ab[k]);
                        sphere_mat = make_shared<metal_simd4>(albedo, afuzz[k]); 
                        world.add(make_shared<sphere>(center, 0.2f, sphere_mat));                       
                    } 
                    else {
                        // Glass material
                        sphere_mat = make_shared<dielectric_simd4>(1.5f);
                        world.add(make_shared<sphere>(center, 0.2f, sphere_mat));
                        
                    }
                   
                }
            }
        }
    }
    // Three large hero spheres
    auto material1 = make_shared<dielectric_simd4>(1.5f);
    world.add(make_shared<sphere>(point3(0.0f, 1.0f, 0.0f), 1.0f, material1));

    auto material2 = make_shared<lambertian_simd4>(color(0.4f, 0.2f, 0.1f));
    world.add(make_shared<sphere>(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

    auto material3 = make_shared<metal_simd4>(color(0.7f, 0.6f, 0.5f), 0.0f);
    world.add(make_shared<sphere>(point3(4.0f, 1.0f, 0.0f), 1.0f, material3));

    auto bvh_root = make_shared<bvh_node4>(world);
    hit_list accelerated_world;
    accelerated_world.add(bvh_root);

    camera cam;
    cam.image_width = user_config.width;
    cam.samples_per_pixel = user_config.samples;
    cam.vfov=20.0f;
    cam.lookFrom=point3(13,2,3);
    cam.lookAt = point3(0,0,0);
    cam.defocus_angle = 0.6f;


    

    std::vector<uint32_t> screen_pixels;

    //Render CPU Ray Tracer & Save Output
    cam.render(accelerated_world, screen_pixels);
    save_image(user_config.filename, cam.image_width, cam.image_height, screen_pixels);

    // Create Window & Upload Texture for Viewing
    GLuint display_texture = 0;
    GLFWwindow* window = create_display_window(cam.image_width, cam.image_height, "Ray Tracer Display", display_texture);
    if (!window) {
        cleanup(nullptr, 0, &vkb_instance);
        return -1;
    }

    update_display_texture(display_texture, cam.image_width, cam.image_height, screen_pixels);
    //  Interactive Display Loop
    while (!glfwWindowShouldClose(window)) {
        display_frame(window, display_texture);
    }
    //  Final Cleanup 
    cleanup(window, display_texture, &vkb_instance);
    return 0;
}