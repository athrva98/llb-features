#include "catch.hpp"
#include "test_helpers.hpp"

using namespace llb_features;

TEST_CASE("LocalGeometryAnalyzer: planar neighborhood", "[geometry]") {
    // Points on a flat XY plane — should have:
    // - smallest eigenvalue ~ 0 (no variation in Z)
    // - high planarity
    // - low sphericity
    auto [points, normals] = makeGrid(5, 0.1f);

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    auto desc = LocalGeometryAnalyzer<float>::computeLocalSurfaceProperties(center, points);

    CHECK(desc.variation < 0.01f);    // nearly zero variation in normal direction
    CHECK(desc.planarity > 0.5f);     // high planarity
    CHECK(desc.sphericity < 0.1f);    // not spherical
    CHECK(desc.stability_score >= 0.0f);
    CHECK(desc.stability_score <= 1.0f);
}

TEST_CASE("LocalGeometryAnalyzer: spherical neighborhood", "[geometry]") {
    // Points on a hemisphere — more isotropic
    auto [points, normals] = makeHemisphere(50, 1.0f);

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    auto desc = LocalGeometryAnalyzer<float>::computeLocalSurfaceProperties(center, points);

    // Hemisphere should have some variation in all directions
    CHECK(desc.variation > 0.0f);
    CHECK(desc.anisotropy < 1.0f);
}

TEST_CASE("LocalGeometryAnalyzer: robust covariance with outliers", "[geometry]") {
    // Flat plane with one extreme outlier
    auto [points, normals] = makeGrid(5, 0.1f);
    // Add outlier far in Z
    points.push_back({0.2f, 0.2f, 100.0f});

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    auto cov = LocalGeometryAnalyzer<float>::computeRobustCovariance(center, points);

    // Huber weighting should suppress the outlier
    // The Z variance should be much less than 100^2
    CHECK(cov(2, 2) < 100.0f);
}

TEST_CASE("LocalGeometryAnalyzer: too few neighbors returns identity", "[geometry]") {
    AlignedVector<float> tiny = {{0, 0, 0}, {1, 0, 0}};
    Eigen::Vector3f query(0.5f, 0.0f, 0.0f);

    auto cov = LocalGeometryAnalyzer<float>::computeRobustCovariance(query, tiny);
    CHECK(cov.isApprox(Eigen::Matrix3f::Identity()));
}

TEST_CASE("LocalGeometryAnalyzer: covariance is symmetric", "[geometry]") {
    auto [points, normals] = makeHemisphere(30, 1.0f);
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    auto cov = LocalGeometryAnalyzer<float>::computeRobustCovariance(center, points);
    CHECK(cov.isApprox(cov.transpose(), 1e-6f));
}

TEST_CASE("LocalGeometryAnalyzer: eigenvalues are non-negative", "[geometry]") {
    auto [points, normals] = makeHemisphere(30, 1.0f);
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& p : points) center += p;
    center /= static_cast<float>(points.size());

    auto cov = LocalGeometryAnalyzer<float>::computeRobustCovariance(center, points);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    auto evals = solver.eigenvalues();
    CHECK(evals(0) >= -1e-7f);
    CHECK(evals(1) >= -1e-7f);
    CHECK(evals(2) >= -1e-7f);
}
