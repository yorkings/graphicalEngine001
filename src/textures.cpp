#include "../header/texture.h"

SRGBTable::SRGBTable() {
    for (int i = 0; i < 256; ++i) {
        float c = i / 255.0f;
        lut[i] = (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }
}

const SRGBTable& SRGBTable::get() {
    static SRGBTable instance;
    return instance;
}


void texture::sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const {
    alignas(16) float u[4], v[4], r[4], g[4], b[4];
    _mm_store_ps(u, u4);
    _mm_store_ps(v, v4);
    for (int i = 0; i < 4; ++i) {
        Color3f c = sample(u[i], v[i]);
        r[i] = c.r; g[i] = c.g; b[i] = c.b;
    }

    out_r = _mm_load_ps(r);
    out_g = _mm_load_ps(g);
    out_b = _mm_load_ps(b);
}



void SolidColor::sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const {
    out_r = _mm_set1_ps(color.r);
    out_g = _mm_set1_ps(color.g);
    out_b = _mm_set1_ps(color.b);
}



Color3f CheckerTexture::sample(float u, float v) const {
    int x = static_cast<int>(std::floor(u * inv_scale));
    int y = static_cast<int>(std::floor(v * inv_scale));
    return ((x ^ y) & 1) ? odd->sample(u, v) : even->sample(u, v);
}



ImageTexture::ImageTexture(const uint8_t* src_pixels, int w, int h, int channels): width(w), height(h) {
    is_pow2 = (w & (w - 1)) == 0 && (h & (h - 1)) == 0;
    if (is_pow2) {
        mask_w = w - 1;
        mask_h = h - 1;
    }
    size_t bytes = static_cast<size_t>(w * h * 4);
    pixels = static_cast<uint8_t*>(_mm_malloc(bytes, 64));
    for (int i = 0; i < w * h; ++i) {
        pixels[i * 4 + 0] = src_pixels[i * channels + 0];
        pixels[i * 4 + 1] = src_pixels[i * channels + 1];
        pixels[i * 4 + 2] = src_pixels[i * channels + 2];
        pixels[i * 4 + 3] = (channels == 4) ? src_pixels[i * channels + 3] : 255;
    }
}

ImageTexture::~ImageTexture() {
    if (pixels) _mm_free(pixels);
}

void ImageTexture::get_coords(int x, int y, int& out_x, int& out_y) const {
    if (is_pow2) {
        out_x = x & mask_w;
        out_y = y & mask_h;
    } else {
        out_x = (x % width + width) % width;
        out_y = (y % height + height) % height;
    }
}

Color3f ImageTexture::fetch_pixel_linear(int x, int y) const {
    int cx, cy;
    get_coords(x, y, cx, cy);

    const uint8_t* p = &pixels[(cy * width + cx) * 4];
    const float* lut = SRGBTable::get().lut;

    return Color3f(lut[p[0]], lut[p[1]], lut[p[2]], p[3] * (1.0f / 255.0f));
}

Color3f ImageTexture::sample(float u, float v) const {
    if (!pixels) return Color3f(1.0f, 0.0f, 1.0f);

    u = u - std::floor(u);
    v = 1.0f - (v - std::floor(v));

    float fu = u * width - 0.5f;
    float fv = v * height - 0.5f;

    int x0 = static_cast<int>(std::floor(fu));
    int y0 = static_cast<int>(std::floor(fv));

    float frac_u = fu - x0;
    float frac_v = fv - y0;

    Color3f c00 = fetch_pixel_linear(x0,     y0);
    Color3f c10 = fetch_pixel_linear(x0 + 1, y0);
    Color3f c01 = fetch_pixel_linear(x0,     y0 + 1);
    Color3f c11 = fetch_pixel_linear(x0 + 1, y0 + 1);

    float w00 = (1.0f - frac_u) * (1.0f - frac_v);
    float w10 = frac_u * (1.0f - frac_v);
    float w01 = (1.0f - frac_u) * frac_v;
    float w11 = frac_u * frac_v;

    return c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11;
}

void ImageTexture::sample_4x(vec4 u4, vec4 v4, vec4& out_r, vec4& out_g, vec4& out_b) const {
    if (!pixels) {
        out_r = _mm_set1_ps(1.0f);
        out_g = _mm_set1_ps(0.0f);
        out_b = _mm_set1_ps(1.0f);
        return;
    }
    alignas(16) float u[4], v[4];
    _mm_store_ps(u, u4);
    _mm_store_ps(v, v4);
    alignas(16) float r[4], g[4], b[4];
    for (int i = 0; i < 4; ++i) {
        Color3f col = sample(u[i], v[i]);
        r[i] = col.r;
        g[i] = col.g;
        b[i] = col.b;
    }

    out_r = _mm_load_ps(r);
    out_g = _mm_load_ps(g);
    out_b = _mm_load_ps(b);
}