#pragma once
#include "general.h"

class vec3{
    private:

    public:
        __m128 v;
        inline vec3():v(_mm_setzero_ps()){}
        inline vec3(float x,float y,float z):v(_mm_set_ps(0.0,z,y,x)){}
        inline vec3(__m128 v):v(v){}
        inline float x()const {return _mm_cvtss_f32(v);}
        inline float y()const{return _mm_cvtss_f32(_mm_shuffle_ps(v,v,_MM_SHUFFLE(1,1,1,1)));}
        inline float z()const{return _mm_cvtss_f32(_mm_shuffle_ps(v,v,_MM_SHUFFLE(2,2,2,2)));}
        //indexing
        inline float operator[](int i)const{ return (i==0) ? x() : ((i==1) ? y():z());}
        inline const float& operator[](int i){ return (i == 0)? x():((i==1)?y() :z()) ;}
        //operations
        inline vec3 operator-()const{ return vec3(_mm_xor_ps(v,_mm_set1_ps(-0.0)));}

        inline  vec3 &operator+=(const vec3 &b){ v=_mm_add_ps(v,b.v);return *this;}

        inline vec3 &operator*=(const float d){
            __m128 d_dat=_mm_set1_ps(d);
            v=_mm_mul_ps(v,d_dat);
            return *this;
        }
        
        inline vec3 &operator/=(const float d){ return *this *= (1.0f/d);}

        inline float length_squared() const{
            __m128 dot_res=_mm_dp_ps(v,v,0x71);//store the result on lane 0
            return _mm_cvtss_f32(dot_res);
        }

        inline float length()const{
            return std::sqrt(length_squared());
        }
};

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(_mm_add_ps(u.v, v.v));
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(_mm_sub_ps(u.v, v.v));
}

inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(_mm_mul_ps(u.v, v.v)); // Component-wise multiplication
}

inline vec3 operator*(float t, const vec3& v) {
    return vec3(_mm_mul_ps(_mm_set1_ps(t), v.v));
}

inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, float t) {
    return (1.0f / t) * v;
}

inline float dot(const vec3& u, const vec3& v) {
    __m128 dot_res = _mm_dp_ps(u.v, v.v, 0x71);
    return _mm_cvtss_f32(dot_res);
}
inline vec3 cross(const vec3 &u, const vec3 &v){
    __m128 u_yzx =_mm_shuffle_ps(u.v,u.v,_MM_SHUFFLE(3,0,2,1));
    __m128 v_zxy =_mm_shuffle_ps(v.v,v.v,_MM_SHUFFLE(3,1,0,2));
    __m128 u_zxy =_mm_shuffle_ps(u.v,u.v,_MM_SHUFFLE(3,1,0,2));    
    __m128 v_yzx =_mm_shuffle_ps(v.v,v.v,_MM_SHUFFLE(3,0,2,1));

    __m128 left  = _mm_mul_ps(u_yzx, v_zxy);
    __m128 right = _mm_mul_ps(u_zxy, v_yzx);

    return vec3(_mm_sub_ps(left, right));

}
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}