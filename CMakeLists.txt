cmake_minimum_required(VERSION 3.20)
project("graphical engine")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

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
set(SOURCE_FILES
    src/camera.cpp
    src/functions.cpp
    src/textures.cpp
)

add_executable(GraphicsRenderer main.cpp ${SOURCE_FILES})
# Apply OpenMP flags and optimization
target_link_libraries(GraphicsRenderer PRIVATE 
    glfw
    vk-bootstrap
    OpenGL::GL
    Vulkan::Vulkan 
    OpenMP::OpenMP_CXX
)

# Required for dynamic symbol loading (Linux/macOS/Unix)
if(UNIX AND NOT APPLE)
    target_link_libraries(GraphicsRenderer PRIVATE ${CMAKE_DL_LIBS})
endif()
if(WIN32)
    target_link_libraries(GraphicsRenderer PRIVATE 
        winmm 
        ws2_32 
        # For MSVC's vcpkg or system libs:
        ${CMAKE_DL_LIBS} 
    )
endif()

if(MSVC)
    target_compile_options(GraphicsRenderer PRIVATE /W4 /MP /permissive- /Zc:__cplusplus)
    target_compile_options(GraphicsRenderer PRIVATE /arch:AVX2 /fp:fast)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(GraphicsRenderer PRIVATE /Od /Zi)
        target_compile_definitions(GraphicsRenderer PRIVATE _DEBUG)
    else()
        target_compile_options(GraphicsRenderer PRIVATE /O2 /Oi /GL)
        target_link_options(GraphicsRenderer PRIVATE /LTCG)
        target_compile_definitions(GraphicsRenderer PRIVATE NDEBUG)
    endif()
else()
    # GCC / Clang
    target_compile_options(GraphicsRenderer PRIVATE -Wall -Wextra)
    target_compile_options(GraphicsRenderer PRIVATE -march=native -mtune=native -ffast-math)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(GraphicsRenderer PRIVATE -Wpedantic -Wshadow -g -O0)
        target_compile_definitions(GraphicsRenderer PRIVATE DEBUG)
    else()
        target_compile_options(GraphicsRenderer PRIVATE -O3 -DNDEBUG)
        target_link_options(GraphicsRenderer PRIVATE -flto)
    endif()
endif()

message(STATUS "==========================================")
message(STATUS " Build Type   : ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ Standard : ${CMAKE_CXX_STANDARD}")
message(STATUS " Compiler ID  : ${CMAKE_CXX_COMPILER_ID}")
if(OpenMP_CXX_FOUND)
    message(STATUS " OpenMP Flags : ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "==========================================")


