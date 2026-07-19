#include "catch.hpp"
#include "test_helpers.hpp"
#include <cmath>

using namespace llb_features;

TEST_CASE("Weights: features are finite for normal-scale cloud", "[weights]") {
    auto [points, normals] = makeHemisphere(50, 1.0f);
    LLBFeatures<float> llb(points, normals, 10, 4, 0.5f);
    auto features = llb.computeFeatures();
    for (const auto& f : features) CHECK(f.allFinite());
}

TEST_CASE("Weights: anisotropic weight uses normal_weight_factor", "[weights]") {
    auto [points, normals] = makeHemisphere(50, 1.0f);

    LLBFeatures<float> llb_zero(points, normals, 10, 4, 0.0f);
    auto feat_zero = llb_zero.computeFeatures();

    LLBFeatures<float> llb_one(points, normals, 10, 4, 1.0f);
    auto feat_one = llb_one.computeFeatures();

    bool any_different = false;
    for (size_t i = 0; i < feat_zero.size(); ++i) {
        if (!feat_zero[i].isApprox(feat_one[i], 1e-4f)) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
}

TEST_CASE("Weights: coincident points produce finite features", "[weights]") {
    AlignedVector<float> points(20, Eigen::Vector3f(1.0f, 2.0f, 3.0f));
    AlignedVector<float> normals(20, Eigen::Vector3f(0.0f, 0.0f, 1.0f));

    // This should not crash with coincident points
    LLBFeatures<float> llb(points, normals, 10, 3, 0.5f);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 20);
    for (const auto& f : features) {
        CHECK(f.allFinite());
    }
}
