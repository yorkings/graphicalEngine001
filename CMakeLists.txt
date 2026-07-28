cmake_minimum_required(VERSION 3.20)
project("graphical engine")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)




# Force executable to the source directory root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)




# Fetch vk-bootstrap automatically
include(FetchContent)
FetchContent_Declare(
    vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        v1.3.279 # Use a specific tag for stability
)

FetchContent_MakeAvailable(vk_bootstrap)

find_package(OpenMP REQUIRED)
# set(Source_files
#     src/renderer.cpp
#     src/physics.cpp
#     src/utils.cpp
# )
add_executable(GraphicsEngine main.cpp)
# Apply OpenMP flags and optimization
target_link_libraries(GraphicsEngine PRIVATE 
    glfw
    vk-bootstrap
    OpenGL::GL
    Vulkan::Vulkan 
    OpenMP::OpenMP_CXX
)
target_compile_options(GraphicsEngine PRIVATE -O3)

# Optional: Add more optimization flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(GraphicsEngine PRIVATE -march=native -mtune=native -ffast-math )
    # Required on Linux/Unix for dynamic symbol loading used by vk-bootstrap
    target_link_libraries(GraphicsEngine PRIVATE ${CMAKE_DL_LIBS})
endif()

# Print status to verify flags
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
endif()
# For debugging (optional)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(GraphicsEngine PRIVATE -g -O0)
else()
    target_compile_options(GraphicsEngine PRIVATE -O3)
endif()



