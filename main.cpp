#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>
#include "VkBootstrap.h"

int main() {
    // 1. Build Headless Vulkan Instance for background compute
    vkb::InstanceBuilder instance_builder;    
    auto instance_ret = instance_builder
        .set_app_name("RayTracerEngine")
        .set_headless()// Tells VkBootstrap this Vulkan instance is for headless compute
#ifndef NDEBUG
        .request_validation_layers(true)
        .use_default_debug_messenger()
#endif
        .build();

    if (!instance_ret) {
        std::cerr << "Failed to create Vulkan instance: " 
                  << instance_ret.error().message() << "\n";
        return -1;
    }

    vkb::Instance vkb_instance = instance_ret.value();
    std::cout << "Vulkan Compute Instance created successfully!\n";

    // 2. Select Physical Device (GPU) for Compute
    vkb::PhysicalDeviceSelector selector{ vkb_instance };
    auto phys_ret = selector
        .set_minimum_version(1, 2)
        .allow_any_gpu_device_type(true)
        .select();

    if (!phys_ret) {
        std::cerr << "Failed to select Vulkan GPU: " << phys_ret.error().message() << "\n";
        vkb::destroy_instance(vkb_instance);
        return -1;
    }
    vkb::PhysicalDevice vkb_physical_device = phys_ret.value();
    std::cout << "Selected GPU: " << vkb_physical_device.name << "\n";

    // 3. Initialize GLFW & OpenGL Window for Display
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        vkb::destroy_instance(vkb_instance);
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Graphical Engine", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        vkb::destroy_instance(vkb_instance);
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 4. Main Render Loop
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 5. Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    vkb::destroy_instance(vkb_instance);

    return 0;
}