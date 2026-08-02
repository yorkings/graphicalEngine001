#include "general.h"
#include "polygons.h"
class camera{
    private:
        vec3 camera_center;
        point3 pixel00_loc;        // Location of top-left pixel (Pixel [0,0])
        vec3   pixel_delta_u;  // Offset to pixel to the right
        vec3   pixel_delta_v;//ofset to pixel below
        //system camera functions
        vec3   u, v, w;            // Camera frame coordinate system
        // Defocus blur (Depth of Field) parameters
        vec3   defocus_disk_u; 
        vec3   defocus_disk_v;

        void initialize(){}
        Ray get_ray(){}
        color ray_color(){}
        
    public:
        float aspect_ratio= 16.0f / 9.0f;
        int   image_width = 400;            
        int   samples_per_pixel  = 10;  
        int   max_depth = 50;
        // View camera geometry
        float vfov     = 90.0f;           // Vertical field-of-view in degrees
        point3 lookFrom= point3(0, 0, 0); // Point camera is looking from
        point3 lookAt  = point3(0, 0, -1);// Point camera is looking at
        vec3   vup     = vec3(0, 1, 0);   // Camera-relative "up" direction  
        // Depth of Field (Lens controls)
        float defocus_angle= 0.0f;// Variation angle of rays through lens
        float focus_dist= 10.0f;// Distance from lookFrom to plane of perfect focus     
        
        void render(){}      
};