#define _USE_MATH_DEFINES
#include <cmath>
#include <llb_features.hpp>
#include <chrono>
#include <iostream>

int main() {
    const int N = 5000;
    llb_features::AlignedVector<float> points, normals;
    float golden = (1.0f + std::sqrt(5.0f)) / 2.0f;
    for (int i = 0; i < N; ++i) {
        float theta = 2.0f * float(M_PI) * i / golden;
        float phi = std::acos(1.0f - float(i) / N);
        phi = std::min(phi, float(M_PI) / 2.0f);
        float x = std::sin(phi) * std::cos(theta);
        float y = std::sin(phi) * std::sin(theta);
        float z = std::cos(phi);
        points.push_back({x, y, z});
        normals.push_back(Eigen::Vector3f(x, y, z).normalized());
    }

    // Warmup
    {
        llb_features::LLBFeatures<float> llb(points, normals, 15, 5, 0.3f);
        auto f = llb.computeFeatures();
    }

    // Benchmark current
    auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 30; ++iter) {
        llb_features::LLBFeatures<float> llb(points, normals, 15, 5, 0.3f);
        auto f = llb.computeFeatures();
        if (f[0].squaredNorm() < -1.0f) std::cout << "x";
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 30.0;
    std::cout << "k=15 e=5 on " << N << " pts: " << ms << " ms/iter\n";
    return 0;
}
