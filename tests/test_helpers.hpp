#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <llb_features.hpp>
#include <vector>

// Generates a point cloud on a hemisphere of given radius with N points.
// Returns both points and outward normals.
inline std::pair<llb_features::AlignedVector<float>, llb_features::AlignedVector<float>>
makeHemisphere(int N, float radius = 1.0f) {
    llb_features::AlignedVector<float> points;
    llb_features::AlignedVector<float> normals;
    points.reserve(N);
    normals.reserve(N);

    // Fibonacci sphere for roughly uniform distribution
    float golden = (1.0f + std::sqrt(5.0f)) / 2.0f;
    for (int i = 0; i < N; ++i) {
        float theta = 2.0f * static_cast<float>(M_PI) * i / golden;
        float phi = std::acos(1.0f - static_cast<float>(i) / N); // [0, pi/2-ish]
        phi = std::min(phi, static_cast<float>(M_PI) / 2.0f);    // upper hemisphere

        float x = radius * std::sin(phi) * std::cos(theta);
        float y = radius * std::sin(phi) * std::sin(theta);
        float z = radius * std::cos(phi);

        Eigen::Vector3f p(x, y, z);
        Eigen::Vector3f n = p.normalized();
        points.push_back(p);
        normals.push_back(n);
    }
    return {points, normals};
}

// Generates a flat grid in the XY plane
inline std::pair<llb_features::AlignedVector<float>, llb_features::AlignedVector<float>>
makeGrid(int side, float spacing = 0.1f) {
    llb_features::AlignedVector<float> points;
    llb_features::AlignedVector<float> normals;
    int N = side * side;
    points.reserve(N);
    normals.reserve(N);

    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            float x = i * spacing;
            float y = j * spacing;
            points.push_back({x, y, 0.0f});
            normals.push_back({0.0f, 0.0f, 1.0f});
        }
    }
    return {points, normals};
}

// Generates a simple point cloud without normals (small cube grid)
inline llb_features::AlignedVector<float> makeCubeGrid(int side, float spacing = 0.1f) {
    llb_features::AlignedVector<float> points;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k)
                points.push_back({i * spacing, j * spacing, k * spacing});
    return points;
}
