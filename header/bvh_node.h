#pragma once
#include "hit_funcs.h"
#include "aabb.h"

class bvh_node4 : public hitable {
private:
    aabb4 child_bounds;                          // SoA: lane i stores bounding box of child i
    aabb4 node_bbox;                             // Broadcasted bounding box of this entire node
    std::array<shared_ptr<hitable>, 4> children; // Child pointers (sub-nodes or leaf primitives)
    int num_children = 0;

    inline vec4 blend_ps(vec4 a, vec4 b, vec4 mask) const { return _mm_blendv_ps(a, b, mask); }

    inline void update_hit_record(hit_rec& dst, const hit_rec& src, vec4 lane_mask) const {
        dst.t = blend_ps(dst.t, src.t, lane_mask);            
        dst.p_x = blend_ps(dst.p_x, src.p_x, lane_mask);
        dst.p_y = blend_ps(dst.p_y, src.p_y, lane_mask);
        dst.p_z = blend_ps(dst.p_z, src.p_z, lane_mask);
        dst.nx = blend_ps(dst.nx, src.nx, lane_mask);
        dst.ny = blend_ps(dst.ny, src.ny, lane_mask);
        dst.nz = blend_ps(dst.nz, src.nz, lane_mask);
        dst.front_face = blend_ps(dst.front_face, src.front_face, lane_mask);

        int mask_bits = _mm_movemask_ps(lane_mask);    
        if (mask_bits & 1) dst.mat[0] = src.mat[0];
        if (mask_bits & 2) dst.mat[1] = src.mat[1];
        if (mask_bits & 4) dst.mat[2] = src.mat[2];
        if (mask_bits & 8) dst.mat[3] = src.mat[3];
    }

public:
    bvh_node4() = default;

    bvh_node4(const hit_list& list): bvh_node4(const_cast<std::vector<shared_ptr<hitable>>&>(list.objects), 0, list.objects.size()) {}

    bvh_node4(std::vector<shared_ptr<hitable>>& objects, size_t start, size_t end) {
        size_t object_span = end - start;
        // Leaf Node Construction (<= 4 objects)
        if (object_span <= 4) {
            num_children = static_cast<int>(object_span);
            vec3 min_pts[4], max_pts[4];

            for (size_t i = 0; i < object_span; ++i) {
                children[i] = objects[start + i];
                aabb4 child_box = children[i]->bounding_box();                
                // Extract scalar endpoints from SoA lane 0 for packing
                min_pts[i] = vec3(child_box.min_x[0], child_box.min_y[0], child_box.min_z[0]);
                max_pts[i] = vec3(child_box.max_x[0], child_box.max_y[0], child_box.max_z[0]);
            }
            // Pack child bounds into SoA layout
            child_bounds = aabb4(min_pts, max_pts, num_children);
            build_node_bbox();
            return;
        }
        // Interior Node Construction (> 4 objects): Calculate Centroid Bounds
        vec3 centroid_min(infinity, infinity, infinity);
        vec3 centroid_max(-infinity, -infinity, -infinity);
        for (size_t i = start; i < end; ++i) {
            aabb4 box = objects[i]->bounding_box();

            vec3 center((box.min_x[0] + box.max_x[0]) * 0.5f,(box.min_y[0] + box.max_y[0]) * 0.5f,(box.min_z[0] + box.max_z[0]) * 0.5f);
            centroid_min = vec3(std::min(centroid_min.x(), center.x()),std::min(centroid_min.y(), center.y()),std::min(centroid_min.z(), center.z()));
            centroid_max = vec3(std::max(centroid_max.x(), center.x()),std::max(centroid_max.y(), center.y()),std::max(centroid_max.z(), center.z()));
        }
        // Select primary splitting axis (longest centroid extent)
        vec3 extent = centroid_max - centroid_min;
        int axis = 0;
        if (extent.y() > extent.x()) axis = 1;
        if (extent.z() > extent[axis]) axis = 2;

        // Sort objects along the chosen axis
        auto comparator = [axis](const shared_ptr<hitable>& a, const shared_ptr<hitable>& b) {
            aabb4 box_a = a->bounding_box();
            aabb4 box_b = b->bounding_box();
            float center_a = (axis == 0) ? (box_a.min_x[0] + box_a.max_x[0]) :
                             (axis == 1) ? (box_a.min_y[0] + box_a.max_y[0]) :
                                           (box_a.min_z[0] + box_a.max_z[0]);
            float center_b = (axis == 0) ? (box_b.min_x[0] + box_b.max_x[0]) :
                             (axis == 1) ? (box_b.min_y[0] + box_b.max_y[0]) :
                                           (box_b.min_z[0] + box_b.max_z[0]);
            return center_a < center_b;
        };

        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        // Partition elements into 4 equal child splits
        size_t mid1 = start + object_span / 4;
        size_t mid2 = start + object_span / 2;
        size_t mid3 = start + (object_span * 3) / 4;

        size_t splits[5] = { start, mid1, mid2, mid3, end };
        vec3 min_pts[4], max_pts[4];

        for (int i = 0; i < 4; ++i) {
            if (splits[i] < splits[i + 1]) {
                children[i] = make_shared<bvh_node4>(objects, splits[i], splits[i + 1]);
                aabb4 child_box = children[i]->bounding_box();

                // Extract global bounds of child sub-tree
                min_pts[i] = vec3(_mm_cvtss_f32(child_box.min_x), 
                                  _mm_cvtss_f32(child_box.min_y), 
                                  _mm_cvtss_f32(child_box.min_z));
                max_pts[i] = vec3(_mm_cvtss_f32(child_box.max_x), 
                                  _mm_cvtss_f32(child_box.max_y), 
                                  _mm_cvtss_f32(child_box.max_z));
                num_children++;
            }
        }

        child_bounds = aabb4(min_pts, max_pts, num_children);
        build_node_bbox();
    }

    aabb4 bounding_box() const override { return node_bbox; }

    bool hit(const Raypackets& r, Interval4 ray_t, hit_rec& rec) const override {
        vec4 node_hits = node_bbox.hit_packet(r, ray_t);
        if (_mm_movemask_ps(node_hits) == 0) return false;

        // Step 2: Test each ray against ALL 4 child boxes simultaneously
        alignas(16) float ox[4], oy[4], oz[4];
        alignas(16) float idx[4], idy[4], idz[4];
        alignas(16) float tmin[4], tmax[4];

        _mm_store_ps(ox, r.orig_x);      _mm_store_ps(oy, r.orig_y);      _mm_store_ps(oz, r.orig_z);
        _mm_store_ps(idx, r.inv_dir_x);  _mm_store_ps(idy, r.inv_dir_y);  _mm_store_ps(idz, r.inv_dir_z);
        _mm_store_ps(tmin, ray_t.min);   _mm_store_ps(tmax, ray_t.max);

        vec4 active_children_mask = _mm_setzero_ps();

        for (int i = 0; i < 4; ++i) {
            if (tmin[i] >= tmax[i]) continue; // Skip inactive ray lanes

            vec4 r_ox   = _mm_set1_ps(ox[i]);
            vec4 r_oy   = _mm_set1_ps(oy[i]);
            vec4 r_oz   = _mm_set1_ps(oz[i]);
            vec4 r_idx  = _mm_set1_ps(idx[i]);
            vec4 r_idy  = _mm_set1_ps(idy[i]);
            vec4 r_idz  = _mm_set1_ps(idz[i]);
            vec4 r_tmin = _mm_set1_ps(tmin[i]);
            vec4 r_tmax = _mm_set1_ps(tmax[i]);

            // Single SSE call intersects ray i with 4 child boxes
            vec4 hit_mask = child_bounds.hit_1ray_vs_4boxes(
                r_ox, r_oy, r_oz, r_idx, r_idy, r_idz, r_tmin, r_tmax
            );

            active_children_mask = _mm_or_ps(active_children_mask, hit_mask);
        }

        int child_bitmask = _mm_movemask_ps(active_children_mask);
        if (child_bitmask == 0) return false;

        // Step 3: Traverse sub-trees of hit child nodes
        hit_rec temp_rec;
        bool hit_anything = false;
        vec4 accumulated_hit_mask = _mm_setzero_ps();

        for (int i = 0; i < num_children; ++i) {
            if (child_bitmask & (1 << i)) {
                if (children[i]->hit(r, ray_t, temp_rec)) {
                    hit_anything = true;
                    accumulated_hit_mask = _mm_or_ps(accumulated_hit_mask, temp_rec.hit_mask);
                    ray_t.max = blend_ps(ray_t.max, temp_rec.t, temp_rec.hit_mask);
                    update_hit_record(rec, temp_rec, temp_rec.hit_mask);
                }
            }
        }

        rec.hit_mask = accumulated_hit_mask;
        return hit_anything;
    }

private:
    void build_node_bbox() {
        // Union child bounds across SIMD registers to form global node bounding box
        float min_x = _mm_cvtss_f32(child_bounds.min_x);
        float min_y = _mm_cvtss_f32(child_bounds.min_y);
        float min_z = _mm_cvtss_f32(child_bounds.min_z);
        float max_x = _mm_cvtss_f32(child_bounds.max_x);
        float max_y = _mm_cvtss_f32(child_bounds.max_y);
        float max_z = _mm_cvtss_f32(child_bounds.max_z);

        alignas(16) float mx[4], my[4], mz[4], Mx[4], My[4], Mz[4];
        _mm_store_ps(mx, child_bounds.min_x); _mm_store_ps(my, child_bounds.min_y); _mm_store_ps(mz, child_bounds.min_z);
        _mm_store_ps(Mx, child_bounds.max_x); _mm_store_ps(My, child_bounds.max_y); _mm_store_ps(Mz, child_bounds.max_z);

        for (int i = 1; i < num_children; ++i) {
            min_x = std::min(min_x, mx[i]);
            min_y = std::min(min_y, my[i]);
            min_z = std::min(min_z, mz[i]);
            max_x = std::max(max_x, Mx[i]);
            max_y = std::max(max_y, My[i]);
            max_z = std::max(max_z, Mz[i]);
        }

        node_bbox = aabb4(vec3(min_x, min_y, min_z), vec3(max_x, max_y, max_z));
    }
};