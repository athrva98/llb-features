#include "catch.hpp"
#include "test_helpers.hpp"

using namespace llb_features;

TEST_CASE("LLBFeatures: basic output dimensions (no normals)", "[core]") {
    auto points = makeCubeGrid(4);  // 64 points
    size_t k = 10;
    size_t eigvecs = 5;
    LLBFeatures<float> llb(points, k, eigvecs);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == points.size());
    for (const auto& f : features) {
        CHECK(f.rows() == static_cast<int>(eigvecs));
    }
}

TEST_CASE("LLBFeatures: basic output dimensions (with normals)", "[core]") {
    auto [points, normals] = makeHemisphere(100);
    size_t k = 20;
    size_t eigvecs = 8;
    LLBFeatures<float> llb(points, normals, k, eigvecs, 0.45f);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == points.size());
    for (const auto& f : features) {
        CHECK(f.rows() == static_cast<int>(eigvecs));
    }
}

TEST_CASE("LLBFeatures: all features are finite", "[core]") {
    auto [points, normals] = makeHemisphere(80);
    LLBFeatures<float> llb(points, normals, 15, 5, 0.3f);
    auto features = llb.computeFeatures();

    for (size_t i = 0; i < features.size(); ++i) {
        REQUIRE(features[i].allFinite());
    }
}

TEST_CASE("LLBFeatures: features are non-trivial (not all zero)", "[core]") {
    auto [points, normals] = makeHemisphere(60);
    LLBFeatures<float> llb(points, normals, 15, 5, 0.3f);
    auto features = llb.computeFeatures();

    int nonzero = 0;
    for (const auto& f : features) {
        if (f.squaredNorm() > 1e-10f) ++nonzero;
    }
    // At least 80% should be nonzero
    CHECK(nonzero >= static_cast<int>(features.size()) * 0.8);
}

TEST_CASE("LLBFeatures: consistent dimensions across all points", "[core]") {
    auto points = makeCubeGrid(4);  // 64 points
    LLBFeatures<float> llb(points, 15, 7);
    auto features = llb.computeFeatures();

    REQUIRE(!features.empty());
    auto dim = features[0].rows();
    CHECK(dim == 7);
    for (const auto& f : features) {
        CHECK(f.rows() == dim);
    }
}

TEST_CASE("LLBFeatures: deterministic output", "[core]") {
    auto [points, normals] = makeGrid(6);  // 36 points
    size_t k = 10;
    size_t eigvecs = 4;

    LLBFeatures<float> llb1(points, normals, k, eigvecs, 0.2f);
    auto feat1 = llb1.computeFeatures();

    LLBFeatures<float> llb2(points, normals, k, eigvecs, 0.2f);
    auto feat2 = llb2.computeFeatures();

    REQUIRE(feat1.size() == feat2.size());
    for (size_t i = 0; i < feat1.size(); ++i) {
        CHECK(feat1[i].isApprox(feat2[i], 1e-5f));
    }
}

TEST_CASE("LLBFeatures: double precision works", "[core]") {
    AlignedVector<double> points;
    for (int i = 0; i < 30; ++i) {
        double x = (i % 5) * 0.1;
        double y = (i / 5) * 0.1;
        double z = 0.0;
        points.push_back({x, y, z});
    }

    LLBFeatures<double> llb(points, 8, 4);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 30);
    for (const auto& f : features) {
        CHECK(f.rows() == 4);
        CHECK(f.allFinite());
    }
}

TEST_CASE("LLBFeatures: minimum viable cloud (6 points, k=3, eigvecs=1)", "[core]") {
    AlignedVector<float> points = {
        {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}
    };
    LLBFeatures<float> llb(points, 3, 1);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 6);
    for (const auto& f : features) {
        CHECK(f.rows() == 1);
        CHECK(f.allFinite());
    }
}

TEST_CASE("LLBFeatures: k_neighbors = cloud size", "[core]") {
    auto points = makeCubeGrid(2);  // 8 points
    LLBFeatures<float> llb(points, 8, 3);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 8);
    for (const auto& f : features) {
        CHECK(f.rows() == 3);
        CHECK(f.allFinite());
    }
}

TEST_CASE("LLBFeatures: normal_weight_factor affects output", "[core]") {
    auto [points, normals] = makeHemisphere(50);

    LLBFeatures<float> llb_low(points, normals, 15, 5, 0.0f);
    auto feat_low = llb_low.computeFeatures();

    LLBFeatures<float> llb_high(points, normals, 15, 5, 2.0f);
    auto feat_high = llb_high.computeFeatures();

    // Features should differ with different normal weights
    bool any_different = false;
    for (size_t i = 0; i < feat_low.size(); ++i) {
        if (!feat_low[i].isApprox(feat_high[i], 1e-4f)) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
}
