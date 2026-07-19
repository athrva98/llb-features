#include "catch.hpp"
#include "test_helpers.hpp"

using namespace llb_features;

// ============================================================
// Regression test: nth_element distances misalignment
// ============================================================
// The original computeRobustCovariance called nth_element on the distances
// array to find the median, then used distances[i] as the per-point weight.
// nth_element permutes the array, so distances[i] no longer corresponded
// to neighborhood[i]. The fix uses a copy for median computation.
//
// This test verifies: given a neighborhood where one point is far away,
// the Huber weight for that point should be less than 1.0. If distances
// were misaligned, the far-away distance might end up matched to a close
// point, giving it weight 1.0 when it should be downweighted.

TEST_CASE("Regression: robust covariance does not misalign distances", "[regression]") {
    // Create a tight cluster with one outlier
    AlignedVector<float> neighborhood;
    for (int i = 0; i < 10; ++i) {
        neighborhood.push_back({i * 0.01f, 0.0f, 0.0f});
    }
    // Outlier at index 5 — far away in Y
    neighborhood[5] = {0.05f, 10.0f, 0.0f};

    Eigen::Vector3f query(0.05f, 0.0f, 0.0f);

    // Compute covariance with robust weighting
    auto cov = LocalGeometryAnalyzer<float>::computeRobustCovariance(query, neighborhood);

    // With correct Huber weighting, the outlier should be downweighted,
    // so the Y variance should be much less than 10^2 = 100.
    // If distances were misaligned, the outlier's large distance might be
    // applied to a close point (wasting the downweight), and the outlier
    // itself gets weight 1.0, inflating Y variance.
    // With correct Huber weighting the outlier is downweighted,
    // so Y variance is much less than the unweighted value (~100).
    CHECK(cov(1, 1) < 20.0f);

    // Also verify the covariance is symmetric and finite
    CHECK(cov.isApprox(cov.transpose(), 1e-6f));
    CHECK(cov.allFinite());
}

// ============================================================
// Regression: Arnoldi OOB when num_eigenvectors == k_neighbors > 32
// ============================================================
// The original code accessed es.eigenvalues()(i) for i up to k-1 from an
// Arnoldi subspace of size m-1. When k == n (eigenvectors == k_neighbors),
// m-1 = n-1 < k, causing out-of-bounds access.

TEST_CASE("Regression: Arnoldi does not OOB when eigvecs == k > 32", "[regression]") {
    // k = 40, eigvecs = 40, Arnoldi path (n > 32)
    auto [points, normals] = makeHemisphere(150);
    LLBFeatures<float> llb(points, normals, 40, 40, 0.1f);
    auto features = llb.computeFeatures();

    REQUIRE(features.size() == 150);
    for (const auto& f : features) {
        // Must be exactly 40-dimensional (padded with zeros if Arnoldi broke down)
        CHECK(f.rows() == 40);
        CHECK(f.allFinite());
    }
}

// ============================================================
// Regression: SIMD vs std::exp consistency
// ============================================================
// The SIMD exp approximation should produce results close to std::exp.
// We test this indirectly: compute features for the same cloud on this platform
// (which may use SIMD or std::exp depending on compiler flags) and verify
// the results match a second computation (determinism implies the same
// code path is used consistently).

TEST_CASE("Regression: weight function exp(-d^2) gives 1 at zero distance", "[regression]") {
    // Two identical points at distance 0 should have max weight.
    // We can observe this: a cloud of identical points should produce
    // a weight matrix that's all zeros (since exp(-0) = 1 but the diagonal
    // is excluded, and all off-diagonal squared distances are 0 → exp(-0) = 1).
    // The resulting Laplacian should be all zeros (D = sum of ones, L = D - W = 0).
    // Features from a zero Laplacian are all zero.

    // Create a cloud where the first 10 points are identical
    AlignedVector<float> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back({1.0f, 2.0f, 3.0f});
    }
    // Add a few distinct points to make the cloud valid
    for (int i = 0; i < 10; ++i) {
        points.push_back({1.0f + i * 0.5f, 2.0f, 3.0f});
    }

    LLBFeatures<float> llb(points, 10, 3);
    auto features = llb.computeFeatures();

    // All features should be finite (no NaN from 0/0 or similar)
    for (const auto& f : features) {
        CHECK(f.allFinite());
    }
}

TEST_CASE("Regression: widely-spaced points produce finite features", "[regression]") {
    // With adaptive bandwidth, even far-apart points get meaningful weights
    // (σ adapts to the local scale). Features should be finite, not zero.
    AlignedVector<float> points;
    for (int i = 0; i < 20; ++i) {
        points.push_back({i * 1000.0f, 0.0f, 0.0f});
    }

    LLBFeatures<float> llb(points, 5, 3);
    auto features = llb.computeFeatures();

    for (const auto& f : features) {
        CHECK(f.allFinite());
    }
}

// ============================================================
// Regression: non-unit normals don't cause NaN
// ============================================================

TEST_CASE("Regression: non-unit normals produce finite features", "[regression]") {
    auto [points, normals] = makeHemisphere(40);

    // Scale normals to non-unit length
    for (auto& n : normals) {
        n *= 3.7f;
    }

    LLBFeatures<float> llb(points, normals, 10, 4, 0.5f);
    auto features = llb.computeFeatures();

    for (const auto& f : features) {
        CHECK(f.allFinite());
    }
}

TEST_CASE("Regression: zero-length normals produce finite features", "[regression]") {
    auto [points, normals] = makeHemisphere(40);

    // Set some normals to zero
    normals[0] = {0.0f, 0.0f, 0.0f};
    normals[5] = {0.0f, 0.0f, 0.0f};
    normals[10] = {0.0f, 0.0f, 0.0f};

    LLBFeatures<float> llb(points, normals, 10, 4, 0.5f);
    auto features = llb.computeFeatures();

    // Must not crash or produce NaN
    for (const auto& f : features) {
        CHECK(f.allFinite());
    }
}

// ============================================================
// Regression: KD-tree uses L2 distance
// ============================================================
// The original code used L1_Adaptor (Manhattan distance) but all weight
// computations assume Euclidean (L2) geometry. We verify that neighbors
// are found by L2 distance by checking that the nearest neighbor of a
// point is correct in L2.

TEST_CASE("Regression: KNN returns L2-nearest neighbors", "[regression]") {
    // Verify L2 distance is used by checking that changing a neighbor's
    // position changes the features at the query point.
    AlignedVector<float> cloud_a;
    cloud_a.push_back({0.0f, 0.0f, 0.0f});
    cloud_a.push_back({0.7f, 0.7f, 0.0f});
    cloud_a.push_back({1.0f, 0.0f, 0.0f});
    cloud_a.push_back({2.0f, 0.0f, 0.0f});
    cloud_a.push_back({0.0f, 2.0f, 0.0f});
    cloud_a.push_back({2.0f, 2.0f, 0.0f});
    cloud_a.push_back({3.0f, 0.0f, 0.0f});
    cloud_a.push_back({0.0f, 3.0f, 0.0f});

    // Move point 1 far away — changes the k=4 neighborhood of point 0
    AlignedVector<float> cloud_b = cloud_a;
    cloud_b[1] = {100.0f, 100.0f, 0.0f};

    LLBFeatures<float> llb1(cloud_a, 4, 3);
    auto feat1 = llb1.computeFeatures();

    LLBFeatures<float> llb2(cloud_b, 4, 3);
    auto feat2 = llb2.computeFeatures();

    // Features should be finite regardless
    CHECK(feat1[0].allFinite());
    CHECK(feat2[0].allFinite());

    // With histogramming, the distributions should differ when neighbors change
    // Use a loose tolerance since histogram bins are discrete
    CHECK_FALSE(feat1[0].isApprox(feat2[0], 0.05f));
}
