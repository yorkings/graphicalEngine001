#pragma once
#include<mdspan>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <ctime>

#include <omp.h>
#include <chrono>

#include<algorithm>
#include <random> 
#include<cmath>
#include<cstdlib>

#include <atomic>
#include<vector>
#include <sstream>

using std::make_shared;
using std::shared_ptr;


const float infinity =std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f; 

std::string get_time_current(){
    auto now =std::chrono::system_clock::now();
    std::time_t now_time= std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    std::ostringstream oss;
    oss<<std::put_time(local_time,"%Y_%m_%d_%H_%M_%S");
    return oss.str();
}
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}
inline float random_float() {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

inline float random_float(float min, float max) {
    return min + (max - min) * random_float();
}