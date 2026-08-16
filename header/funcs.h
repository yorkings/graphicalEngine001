#pragma once
#include "ray.h"
#include "hit_funcs.h"
#include "polygons.h"
#include "camera.h"
#include "bvh_node.h"

void bouncing_spheres(hit_list &world,camera &cam);
void quads(hit_list &world,camera &cam);