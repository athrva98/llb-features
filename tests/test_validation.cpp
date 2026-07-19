#include "catch.hpp"
#include "test_helpers.hpp"

using namespace llb_features;

TEST_CASE("Validation: point cloud too small", "[validation]") {
    AlignedVector<float> tiny = {{0,0,0}, {1,0,0}, {0,1,0}};
    CHECK_THROWS_AS(
        LLBFeatures<float>(tiny, 3, 1),
        std::invalid_argument);
}

TEST_CASE("Validation: empty point cloud", "[validation]") {
    AlignedVector<float> empty;
    CHECK_THROWS_AS(
        LLBFeatures<float>(empty, 3, 1),
        std::invalid_argument);
}

TEST_CASE("Validation: k_neighbors too small", "[validation]") {
    auto points = makeCubeGrid(3);
    CHECK_THROWS_AS(
        LLBFeatures<float>(points, 2, 1),
        std::invalid_argument);
}

TEST_CASE("Validation: k_neighbors larger than cloud", "[validation]") {
    auto points = makeCubeGrid(2);
    CHECK_THROWS_AS(
        LLBFeatures<float>(points, 100, 5),
        std::invalid_argument);
}

TEST_CASE("Validation: num_eigenvectors > k_neighbors", "[validation]") {
    auto points = makeCubeGrid(3);
    CHECK_THROWS_AS(
        LLBFeatures<float>(points, 10, 15),
        std::invalid_argument);
}

TEST_CASE("Validation: num_eigenvectors = 0", "[validation]") {
    auto points = makeCubeGrid(3);
    CHECK_THROWS_AS(
        LLBFeatures<float>(points, 10, 0),
        std::invalid_argument);
}

TEST_CASE("Validation: normals size mismatch", "[validation]") {
    auto [points, normals] = makeGrid(5);
    normals.pop_back();
    CHECK_THROWS_AS(
        (LLBFeatures<float>(points, normals, 10, 5, 0.1f)),
        std::invalid_argument);
}

TEST_CASE("Validation: Nystrom num_samples < 2", "[validation]") {
    auto points = makeCubeGrid(3);
    CHECK_THROWS_AS(
        (LLBFeatures<float>(points, 10, 5, true, std::optional<size_t>(1))),
        std::invalid_argument);
}

TEST_CASE("Validation: valid inputs do not throw", "[validation]") {
    auto points = makeCubeGrid(3);
    CHECK_NOTHROW(LLBFeatures<float>(points, 10, 5));

    auto [pts, norms] = makeGrid(5);
    CHECK_NOTHROW((LLBFeatures<float>(pts, norms, 10, 5, 0.45f)));
}
