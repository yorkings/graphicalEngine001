#pragma once
#include "../general.h"
#include <GLFW/glfw3.h>
#include "VkBootstrap.h"
#include <fstream>
#include <filesystem>

struct RenderQueries {
    std::string filename;
    int width =1500;
    int samples = 100;
};

RenderQueries fetch_user_queries() {
    RenderQueries queries;    
    std::cout << "Enter save file name (without extension): ";
    if (!(std::cin >> queries.filename) || queries.filename.empty()) {
        std::cerr << "Invalid input. Defaulting filename to 'render'.\n";
        queries.filename = "render";
    }
    std::filesystem::create_directories("image_data");
    queries.filename="image_data/" +queries.filename +"_"+get_time_current();
    return queries;
}
// 0. INITIALIZE VULKAN COMPUTE (HEADLESS)
bool init_vulkan_compute(vkb::Instance& vkb_instance) {
    vkb::InstanceBuilder instance_builder;    
    auto instance_ret = instance_builder
        .set_app_name("RayTracerEngine")
        .set_headless() // Headless compute instance
#ifndef NDEBUG
        .request_validation_layers(true)
        .use_default_debug_messenger()
#endif
        .build();

    if (!instance_ret) {
        std::cerr << "Failed to create Vulkan instance: "<< instance_ret.error().message() << "\n";
        return false;
    }
    vkb_instance = instance_ret.value();
    std::cout << "Vulkan Compute Instance created successfully!\n";
    vkb::PhysicalDeviceSelector selector{ vkb_instance };
    auto phys_ret = selector.set_minimum_version(1, 2).allow_any_gpu_device_type(true).select();
    if (!phys_ret) {
        std::cerr << "Failed to select Vulkan GPU: " << phys_ret.error().message() << "\n";
        vkb::destroy_instance(vkb_instance);
        return false;
    }
    vkb::PhysicalDevice vkb_physical_device = phys_ret.value();
    std::cout << "Selected GPU: " << vkb_physical_device.name << "\n";
    return true;
}

//1. CREATING DISPLAY WINDOW
GLFWwindow* create_display_window(int width, int height, const char* title, GLuint& out_texture_id) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }
    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    // Create & configure display texture
    glGenTextures(1, &out_texture_id);
    glBindTexture(GL_TEXTURE_2D, out_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_TEXTURE_2D);
    return window;
}

// 2. UPLOAD PIXELS TO DISPLAY TEXTURE
void update_display_texture(GLuint texture_id, int width, int height, const std::vector<uint32_t>& screen_pixels) {
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, screen_pixels.data());
}
// 3 DISPLAY FRAME (RENDER QUAD TO WINDOW)
void display_frame(GLFWwindow* window, GLuint texture_id) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
}

// 4. SAVE RENDERED BUFFER TO FILE (.PPM)
void save_image(const std::string& filename, int width, int height, const std::vector<uint32_t>& pixels) {
    std::string full_name = filename + ".ppm";
    std::ofstream file(full_name,std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << full_name << "\n";
        return;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    // Convert RGBA uint32_t to raw binary RGB byte stream
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(pixels.data());
    std::vector<uint8_t> rgb_buffer(width * height * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        rgb_buffer[i * 3 + 0] = byte_ptr[i * 4 + 0]; // Red
        rgb_buffer[i * 3 + 1] = byte_ptr[i * 4 + 1]; // Green
        rgb_buffer[i * 3 + 2] = byte_ptr[i * 4 + 2]; // Blue
    }
    file.write(reinterpret_cast<const char*>(rgb_buffer.data()), rgb_buffer.size());
    std::cout << "Successfully saved image to: " << full_name << "\n";
}
// 5. CLEANUP RESOURCES
void cleanup(GLFWwindow* window, GLuint texture_id, vkb::Instance* vkb_instance = nullptr) {
    if (texture_id) {
        glDeleteTextures(1, &texture_id);
    }
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    if (vkb_instance) {
        vkb::destroy_instance(*vkb_instance);
    }
    std::cout << "Cleaned up all graphics resources.\n";
}