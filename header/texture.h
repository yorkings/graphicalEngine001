#pragma once
#include "vec3_simd4.h"


struct alignas(16) Color3f {
    float r, g, b, a;

    Color3f() : r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}
    Color3f(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}
    Color3f(vec4 v) { _mm_store_ps(&r, v); }

    inline Color3f operator*(float s) const { return Color3f(r * s, g * s, b * s, a); }
    inline Color3f operator+(const Color3f& c) const { return Color3f(r + c.r, g + c.g, b + c.b, a); }
};

class SRGBTable {
    public:
        float lut[256];
        SRGBTable();
        static const SRGBTable& get();
};

struct texture {
    virtual ~texture() = default;

    virtual Color3f sample(float u, float v) const = 0;
    virtual void sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const;
};

class SolidColor : public texture {
    public:
        SolidColor(const Color3f& c) : color(c) {}
        SolidColor(float r, float g, float b) : color(r, g, b) {}

        Color3f sample(float u, float v) const override { return color; }
        void sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const override;

    private:
        Color3f color;
};



class CheckerTexture : public texture {
    public:
        CheckerTexture(float scale, std::shared_ptr<texture> even, std::shared_ptr<texture> odd)
            : even(even), odd(odd), inv_scale(1.0f / scale) {}

        Color3f sample(float u, float v) const override;

    private:
        std::shared_ptr<texture> even;
        std::shared_ptr<texture> odd;
        float inv_scale;
};



class ImageTexture : public texture {
    public:
        ImageTexture() = default;
        ImageTexture(const uint8_t* src_pixels, int w, int h, int channels = 4);
        ~ImageTexture() override;
    
        Color3f sample(float u, float v) const override;
        void sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const override;
    
    private:
        uint8_t* pixels = nullptr;
        int width = 0;
        int height = 0;
        int mask_w = 0;
        int mask_h = 0;
        bool is_pow2 = false;
    
        void get_coords(int x, int y, int& out_x, int& out_y) const;
        Color3f fetch_pixel_linear(int x, int y) const;
};