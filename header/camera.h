#include "general.h"
#include "ray.h"

class camera{
    private:
        vec3 camera_center;
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;
        void initialize(){}
        Ray get_ray(){}
        color ray_color(){}
        
    public:
        float aspect_ratio;
        float focal_length;
        float img_width;
        float img_height;
        float pixel_sample_scale;
        float vfov=90;
        point3 lookFrom=vec3(0.0f,0.0f,0.0f);
        point3 lookAt=point3(0,0,-1);  
        void render(){}      
};