#include "header/general.h"
#include "header/polygons.h"
#include "header/camera.h"
#include "header/bvh_node.h"
#include "header/utils/helpers.h"
#include "header/funcs.h"

int main() {
    RenderQueries user_config = fetch_user_queries();

    vkb::Instance vkb_instance;
    if (!init_vulkan_compute(vkb_instance)) {
        return -1;
    }
    // Setup Scene & Camera
    hit_list world,accelerated_world;   

    camera cam;
    cam.image_width = user_config.width;
    cam.samples_per_pixel = user_config.samples;
    switch (2){
        case 1:bouncing_spheres(world,cam);  break;   
        case 2: quads(world,cam);break;     
        default:
            break;
    }
    auto bvh_root = make_shared<bvh_node4>(world);
    accelerated_world.add(bvh_root);

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