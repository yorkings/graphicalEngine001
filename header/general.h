#pragma once
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
#include<cmath>
#include<cstdlib>

#include <atomic>
#include<vector>

using std::make_shared;
using std::shared_ptr;


const float infinity =std::numeric_limits<float>::infinity();
 

std::string get_time_current(){
    auto now =std::chrono::system_clock::now();
    std::time_t now_time= std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    std::ostringstream oss;
    oss<<std::put_time(local_time,"%Y_%m_%d_%H_%M_%S");
    return oss.str();
}