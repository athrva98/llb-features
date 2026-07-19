#include "catch.hpp"
#include "test_helpers.hpp"

using namespace llb_features;

TEST_CASE("Spectral: small cloud uses direct eigendecomp (n <= 32)", "[spectral]") {
    // With k=10, the Laplacian is 10x10, well under the 32 threshold
    auto points = makeCubeGrid(3);  // 27 points
    LLBFeatures<float> llb(points, 10, 5);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 27);
    for (const auto& f : features) {
        CHECK(f.rows() == 5);
        CHECK(f.allFinite());
    }
}

TEST_CASE("Spectral: large k triggers Arnoldi path (n > 32)", "[spectral]") {
    // With k=40, the Laplacian is 40x40, triggering Arnoldi
    auto [points, normals] = makeHemisphere(200);
    LLBFeatures<float> llb(points, normals, 40, 8, 0.2f);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 200);
    for (const auto& f : features) {
        CHECK(f.rows() == 8);
        CHECK(f.allFinite());
    }
}

TEST_CASE("Spectral: eigvecs = k_neighbors (boundary)", "[spectral]") {
    auto points = makeCubeGrid(3);  // 27 points
    // k=10, eigvecs=10: this is the boundary where Arnoldi OOB used to happen
    LLBFeatures<float> llb(points, 10, 10);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 27);
    for (const auto& f : features) {
        CHECK(f.rows() == 10);
        CHECK(f.allFinite());
    }
}

TEST_CASE("Spectral: eigvecs = k = large triggers Arnoldi boundary", "[spectral]") {
    // k=40, eigvecs=40: Arnoldi path with k == n
    // This was the exact OOB bug in the original code
    auto [points, normals] = makeHemisphere(150);
    LLBFeatures<float> llb(points, normals, 40, 40, 0.1f);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 150);
    for (const auto& f : features) {
        CHECK(f.rows() == 40);
        CHECK(f.allFinite());
    }
}

TEST_CASE("Spectral: single eigenvector", "[spectral]") {
    auto points = makeCubeGrid(3);
    LLBFeatures<float> llb(points, 8, 1);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 27);
    for (const auto& f : features) {
        CHECK(f.rows() == 1);
        CHECK(f.allFinite());
    }
}

TEST_CASE("Spectral: features change with different k_neighbors", "[spectral]") {
    auto points = makeCubeGrid(4);  // 64 points

    LLBFeatures<float> llb_small(points, 5, 3);
    auto feat_small = llb_small.computeFeatures();

    LLBFeatures<float> llb_large(points, 20, 3);
    auto feat_large = llb_large.computeFeatures();

    // Different neighborhoods should produce different features
    bool any_different = false;
    for (size_t i = 0; i < feat_small.size(); ++i) {
        if (!feat_small[i].isApprox(feat_large[i], 1e-4f)) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
}
