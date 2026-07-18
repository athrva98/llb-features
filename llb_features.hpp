/*
 * LLB Features: Local Laplace-Beltrami Features Library for Point Clouds
 *
 * Copyright (c) 2024 Athrva Pandhare
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This software uses the following third-party libraries:
 * - Eigen (MPL2 License): http://eigen.tuxfamily.org
 * - nanoflann (BSD License): https://github.com/jlblancoc/nanoflann
 *
 */

#ifndef LLB_FEATURES_HPP
#define LLB_FEATURES_HPP

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#include <memory>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <nanoflann.hpp>
#include <optional>

namespace llb_features {

// Determine the optimal alignment at compile-time
constexpr size_t determine_optimal_alignment() {
#if defined(__AVX512F__)
    return 64;  // AVX-512 supports 512-bit (64-byte) vectors
#elif defined(__AVX__)
    return 32;  // AVX supports 256-bit (32-byte) vectors
#else
    return 16;  // SSE supports 128-bit (16-byte) vectors
#endif
}

constexpr size_t OPTIMAL_ALIGN = determine_optimal_alignment();

template <typename T>
using AlignedVector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using AlignedVector = std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>;

template <typename T>
class LocalGeometryAnalyzer {
public:
    struct LocalSurfaceDescriptor {
        T variation;                        // Surface variation
        T planarity;                        // Planarity measure
        T anisotropy;                       // Local anisotropy
        T sphericity;                       // Spherical tendency
        AlignedVector3<T> principal_dir;    // Principal direction
        Eigen::Matrix<T, 3, 3> covariance;  // Local covariance
        T stability_score;                  // Feature stability estimate

        LocalSurfaceDescriptor() {
            variation = 0.0;
            planarity = 0.0;
            anisotropy = 0.0;
            sphericity = 0.0;
            principal_dir = {0, 0, 0};
            covariance = Eigen::Matrix<T, 3, 3>::Identity();
            stability_score = 0.0;
        }
    };

    static LocalSurfaceDescriptor computeLocalSurfaceProperties(
            const AlignedVector3<T>& point,
            const std::vector<AlignedVector3<T>,
             Eigen::aligned_allocator<AlignedVector3<T>>>& neighborhood) {
        LocalSurfaceDescriptor desc;

        // Compute robust covariance
        desc.covariance = computeRobustCovariance(point, neighborhood);

        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> solver(desc.covariance);
        if (solver.info() != Eigen::Success) {
            return desc;
        }
        auto eigenvalues = solver.eigenvalues();
        auto eigenvectors = solver.eigenvectors();

        // Sort eigenvalues in ascending order
        if (eigenvalues(1) < eigenvalues(0)) {
            std::swap(eigenvalues(1), eigenvalues(0));
            eigenvectors.col(1).swap(eigenvectors.col(0));
        }
        if (eigenvalues(2) < eigenvalues(1)) {
            std::swap(eigenvalues(2), eigenvalues(1));
            eigenvectors.col(2).swap(eigenvectors.col(1));
            if (eigenvalues(1) < eigenvalues(0)) {
                std::swap(eigenvalues(1), eigenvalues(0));
                eigenvectors.col(1).swap(eigenvectors.col(0));
            }
        }

        T sum_eigenvalues = eigenvalues.sum();
        if (sum_eigenvalues < epsilon) {
            return desc; // Default initialized values
        }

        // Compute surface properties
        desc.variation = eigenvalues(0) / (epsilon + sum_eigenvalues);
        desc.planarity = (eigenvalues(1) - eigenvalues(0)) / (epsilon + eigenvalues(2));
        desc.anisotropy = (eigenvalues(2) - eigenvalues(0)) / (epsilon + eigenvalues(2));
        desc.sphericity = eigenvalues(0) / (epsilon + eigenvalues(2));
        desc.principal_dir = eigenvectors.col(2);

        // Compute stability score based on eigenvalue ratios
        T e1_ratio = eigenvalues(1) / (epsilon + eigenvalues(2));
        T e0_ratio = eigenvalues(0) / (epsilon + eigenvalues(1));
        desc.stability_score = std::abs(e1_ratio - e0_ratio) / (epsilon + 1 + e1_ratio + e0_ratio);

        return desc;
    }

    static Eigen::Matrix<T, 3, 3> computeRobustCovariance(
            const AlignedVector3<T>& query,
            const std::vector<AlignedVector3<T>,
             Eigen::aligned_allocator<AlignedVector3<T>>>& neighborhood) {
        if (neighborhood.size() < MIN_NEIGHBORS) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        }

        // Compute weighted covariance using Huber weights
        Eigen::Matrix<T, 3, 3> covariance = Eigen::Matrix<T, 3, 3>::Zero();
        std::vector<T> distances;
        distances.reserve(neighborhood.size());

        // First pass: compute distances
        for (const auto& point : neighborhood) {
            distances.push_back((point - query).norm());
        }

        // Find median on a COPY — nth_element permutes the array, which would
        // misalign distances[i] with neighborhood[i] in the weight computation below.
        std::vector<T> dist_for_median(distances);
        size_t mid = dist_for_median.size() / 2;
        std::nth_element(dist_for_median.begin(), dist_for_median.begin() + static_cast<std::ptrdiff_t>(mid), dist_for_median.end());
        T median_dist = dist_for_median[mid];
        T huber_threshold = std::max<T>(epsilon, 1.345 * median_dist); // Huber's robust estimator constant

        // Second pass: compute weighted covariance
        T total_weight = 0;
        for (size_t i = 0; i < neighborhood.size(); ++i) {
            AlignedVector3<T> centered = neighborhood[i] - query;
            T dist = distances[i];

            // Huber weight
            T weight = (dist < huber_threshold) ?
                      1.0 : (huber_threshold / (epsilon + dist));

            covariance += weight * (centered * centered.transpose());
            total_weight += weight;
        }

        return covariance / (epsilon + total_weight);
    }

private:
    static constexpr T epsilon = static_cast<T>(1e-10);
    static constexpr size_t MIN_NEIGHBORS = 6;

};

template <typename T>
class LLBPointCloud {
public:
    const AlignedVector<T>* points_ = nullptr;
    void setPoints(const AlignedVector<T>* p) { points_ = p; }

    inline size_t kdtree_get_point_count() const { return points_ ? points_->size() : 0; }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        return (*points_)[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <typename T>
class LLBFeatures {
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
public:
    /**
    * @brief Constructs LLBFeatures with optional fast approximation
    * @param point_cloud Input point cloud data
    * @param k_neighbors Number of neighbors for local analysis
    * @param num_eigenvectors Number of eigenvectors to compute
    * @param use_fast_approximation Whether to use fast approximation method
    * @param num_samples Number of samples to use when fast_approximation is enabled
    *                    (ignored when fast_approximation is false)
    * @throws std::invalid_argument if input parameters are invalid
    */
    LLBFeatures(const AlignedVector<T>& point_cloud,
                size_t k_neighbors = 20,
                size_t num_eigenvectors = 10,
                bool use_fast_approximation = false,
                std::optional<size_t> num_samples = std::nullopt)
        : point_cloud_(point_cloud),
          k_neighbors_(k_neighbors),
          num_eigenvectors_(num_eigenvectors),
          use_fast_approximation_(use_fast_approximation),
          num_samples_(use_fast_approximation ? num_samples.value_or(100) : 0) {
        validateInputs();
        buildKDTree();
    }

    /**
    * @brief Constructs LLBFeatures with optional fast approximation
    * @param point_cloud Input point cloud data
    * @param normals Input point cloud normals data
    * @param k_neighbors Number of neighbors for local analysis
    * @param num_eigenvectors Number of eigenvectors to compute
    * @param use_fast_approximation Whether to use fast approximation method
    * @param num_samples Number of samples to use when fast_approximation is enabled
    *                    (ignored when fast_approximation is false)
    * @throws std::invalid_argument if input parameters are invalid
    */
    LLBFeatures(const AlignedVector<T>& point_cloud,
                const AlignedVector<T>& normals,
                size_t k_neighbors = 20,
                size_t num_eigenvectors = 10,
                const T normal_weight_factor = 0.1,
                bool use_fast_approximation = false,
                std::optional<size_t> num_samples = std::nullopt)
        : point_cloud_(point_cloud),
          normals_(normals),
          k_neighbors_(k_neighbors),
          num_eigenvectors_(num_eigenvectors),
          normal_weight_factor_(normal_weight_factor),
          use_fast_approximation_(use_fast_approximation),
          num_samples_(use_fast_approximation ? num_samples.value_or(100) : 0) {
        validateInputs();
        buildKDTree();
    }

    /**
    * @brief Compute features on a sparse subset, interpolate to full cloud.
    * @param subsample_ratio Fraction of points to compute exactly [0.01, 1.0]
    * @param k_interp Number of nearest anchors for interpolation (>= 1)
    */
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> computeFeaturesSubsampled(
            double subsample_ratio = 0.2, int k_interp = 3) {

        const size_t n = point_cloud_.size();
        subsample_ratio = std::max(0.01, std::min(1.0, subsample_ratio));
        if (subsample_ratio >= 1.0) return computeFeatures();

        size_t num_anchors = std::max<size_t>(
            k_neighbors_,
            static_cast<size_t>(std::ceil(n * subsample_ratio)));
        num_anchors = std::min(num_anchors, n);

        // Deterministic anchor selection
        std::vector<size_t> all_idx(n);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(all_idx.begin(), all_idx.end(), rng);

        std::vector<size_t> anchor_idx(all_idx.begin(),
            all_idx.begin() + static_cast<std::ptrdiff_t>(num_anchors));
        std::vector<bool> is_anchor(n, false);
        for (size_t idx : anchor_idx) is_anchor[idx] = true;

        // Compute exact features for anchors via a temporary LLBFeatures
        AlignedVector<T> anchor_pts(num_anchors);
        for (size_t i = 0; i < num_anchors; ++i)
            anchor_pts[i] = point_cloud_[anchor_idx[i]];

        size_t ak = std::min(k_neighbors_, num_anchors);
        size_t ae = std::min(num_eigenvectors_, ak);

        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> anchor_feats;
        if (!normals_.empty()) {
            AlignedVector<T> anchor_nrm(num_anchors);
            for (size_t i = 0; i < num_anchors; ++i)
                anchor_nrm[i] = normals_[anchor_idx[i]];
            LLBFeatures<T> allb(anchor_pts, anchor_nrm, ak, ae, normal_weight_factor_,
                                 use_fast_approximation_,
                                 use_fast_approximation_ ? std::optional<size_t>(num_samples_) : std::nullopt);
            anchor_feats = allb.computeFeatures();
        } else {
            LLBFeatures<T> allb(anchor_pts, ak, ae, use_fast_approximation_,
                                 use_fast_approximation_ ? std::optional<size_t>(num_samples_) : std::nullopt);
            anchor_feats = allb.computeFeatures();
        }

        int feat_dim = anchor_feats.empty() ? static_cast<int>(num_eigenvectors_)
                                             : static_cast<int>(anchor_feats[0].rows());

        // Build KD-tree on anchor positions for interpolation
        LLBPointCloud<T> anchor_adaptor;
        anchor_adaptor.setPoints(&anchor_pts);
        auto anchor_tree = std::make_unique<KDTree_>(
            3, anchor_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        anchor_tree->buildIndex();

        // Output
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> features(n);
        for (size_t i = 0; i < num_anchors; ++i)
            features[anchor_idx[i]] = anchor_feats[i];

        // Interpolate non-anchors
        int ki = std::min(k_interp, static_cast<int>(num_anchors));

        #pragma omp parallel
        {
            std::vector<size_t> nn_idx(ki);
            std::vector<T> nn_dist(ki);

            #pragma omp for schedule(dynamic, 256)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                if (is_anchor[static_cast<size_t>(i)]) continue;

                nanoflann::KNNResultSet<T> result(ki);
                result.init(nn_idx.data(), nn_dist.data());
                anchor_tree->findNeighbors(result, point_cloud_[i].data(),
                                           nanoflann::SearchParameters());

                Eigen::Matrix<T, Eigen::Dynamic, 1> f =
                    Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(feat_dim);
                T total_w = T(0);
                for (int j = 0; j < ki; ++j) {
                    T w = T(1) / (std::sqrt(nn_dist[j]) + static_cast<T>(1e-10));
                    f += w * anchor_feats[nn_idx[j]];
                    total_w += w;
                }
                features[static_cast<size_t>(i)] = f / total_w;
            }
        }

        return features;
    }

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> computeFeatures() {
        const auto n = static_cast<std::ptrdiff_t>(point_cloud_.size());
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> features(point_cloud_.size());

        #pragma omp parallel
        {
            // Per-thread scratch buffers — eliminates per-point heap allocations
            std::vector<size_t> neighbor_indices(k_neighbors_);
            std::vector<T> distances(k_neighbors_);
            std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>> local_cloud(k_neighbors_);
            std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>> local_normals(
                normals_.empty() ? 0 : k_neighbors_);

            #pragma omp for schedule(dynamic, 64)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                features[static_cast<size_t>(i)] = computeLocalFeaturesFast(
                    static_cast<size_t>(i),
                    neighbor_indices, distances, local_cloud, local_normals);
            }
        }

        return features;
    }

private:
    const AlignedVector<T> point_cloud_;
    const AlignedVector<T> normals_ = {};
    size_t k_neighbors_;
    T normal_weight_factor_;
    size_t num_eigenvectors_;
    bool use_fast_approximation_;
    size_t num_samples_;
    LLBPointCloud<T> pc_adaptor_;
    using KDTree_ = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, LLBPointCloud<T>>, LLBPointCloud<T>, 3>;
    std::unique_ptr<KDTree_> kdtree_;

    void validateInputs() {
        if (point_cloud_.size() < 6) {
            throw std::invalid_argument("Point cloud is empty");
        }
        if (normals_.size() > 0 && normals_.size() != point_cloud_.size()) {
            throw std::invalid_argument("normals.size() must match point.size()");
        }
        if (k_neighbors_ < 3) {
            throw std::invalid_argument("k_neighbors must be at least 3");
        }
        if (num_eigenvectors_ < 1) {
            throw std::invalid_argument("num_eigenvectors must be at least 1");
        }
        if (k_neighbors_ > point_cloud_.size()) {
            throw std::invalid_argument("k_neighbors cannot be larger than the point cloud size");
        }
        if (num_eigenvectors_ > k_neighbors_) {
            throw std::invalid_argument("num_eigenvectors cannot be larger than k_neighbors");
        }
        if (use_fast_approximation_ && num_samples_ < 2) {
            throw std::invalid_argument("num_samples must be at least 2 when using fast approximation");
        }
    }

    void buildKDTree() {
        pc_adaptor_.setPoints(&point_cloud_);
        kdtree_ = std::make_unique<KDTree_>(
            3, pc_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree_->buildIndex();
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> computeLocalFeaturesFast(
            size_t point_idx,
            std::vector<size_t>& neighbor_indices,
            std::vector<T>& distances,
            std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud,
            std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_normals) {

        nanoflann::KNNResultSet<T> resultSet(k_neighbors_);
        resultSet.init(neighbor_indices.data(), distances.data());
        kdtree_->findNeighbors(resultSet, point_cloud_[point_idx].data(), nanoflann::SearchParameters());

        AlignedVector3<T> center = AlignedVector3<T>::Zero();
        for (size_t i = 0; i < k_neighbors_; ++i) {
            local_cloud[i] = point_cloud_[neighbor_indices[i]];
            center += local_cloud[i];
        }
        center /= static_cast<T>(k_neighbors_);

        // computeLocalSurfaceProperties computes the robust covariance internally,
        // so we don't invoke computeRobustCovariance separately here.
        auto desc = LocalGeometryAnalyzer<T>::computeLocalSurfaceProperties(center, local_cloud);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> L;

        if (!normals_.empty()) {
            for (size_t i = 0; i < k_neighbors_; ++i) {
                local_normals[i] = normals_[neighbor_indices[i]];
            }
            L = computeLaplaceBeltrami(local_cloud, local_normals);
        } else {
            L = computeLaplaceBeltrami(local_cloud);
        }
        return computeFastSpectralFeatures(L, num_eigenvectors_) * (1 + desc.stability_score);
    }

    // Combined approach with adaptive blocks and thresholding
    MatrixXT computeEfficientWeights(
            const std::vector<AlignedVector3<T>,
            Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud,
            const std::vector<AlignedVector3<T>,
            Eigen::aligned_allocator<AlignedVector3<T>>>& local_normals) const {

        MatrixXT W = MatrixXT::Zero(k_neighbors_, k_neighbors_);

        const int kn = static_cast<int>(k_neighbors_);

        // Local scale = distance to the scale_k-th nearest neighbor (capped at 20).
        // Each point has only k_neighbors_ - 1 other neighbors, so clamp to that
        // to avoid indexing one past the end of `dists`.
        int scale_k = std::min<int>(20, kn - 1);
        std::vector<T> knn_distances(k_neighbors_);

        // No OpenMP here — this is called from inside the per-point parallel loop.
        // Nested parallelism causes thread explosion.
        for (int i = 0; i < kn; ++i) {
            std::vector<T> dists;
            dists.reserve(k_neighbors_);
            for (int j = 0; j < kn; ++j) {
                if (i != j) {
                    dists.push_back((local_cloud[i] - local_cloud[j]).squaredNorm());
                }
            }
            std::nth_element(dists.begin(), dists.begin() + scale_k - 1, dists.end());
            knn_distances[i] = std::sqrt(dists[scale_k - 1]);
        }

        // Use median of k-nn distances for scale
        std::nth_element(knn_distances.begin(),
                        knn_distances.begin() + static_cast<std::ptrdiff_t>(k_neighbors_/2),
                        knn_distances.end());
        T scale = knn_distances[k_neighbors_/2];
        T squared_scale = scale * scale;

        constexpr int BLOCK_SIZE = 32;

        for (int i = 0; i < kn; ++i) {
            for (int j = i + 1; j < kn; j += BLOCK_SIZE) {
                int block_end = std::min<int>(j + BLOCK_SIZE, kn);

                // Compute weights for block with anisotropic weighting
                for (int k = 0; k < block_end - j; ++k) {
                    const auto& p1 = local_cloud[i];
                    const auto& p2 = local_cloud[j + k];
                    const auto& n1 = local_normals[i];
                    const auto& n2 = local_normals[j + k];

                    T dist = (p1 - p2).squaredNorm();
                    if (dist < squared_scale * static_cast<T>(9.0)) {
                        T weight = computeAnisotropicWeight(
                            p1, p2, n1, n2, normal_weight_factor_);
                        W(i, j + k) = W(j + k, i) = weight;
                    }
                }
            }
        }
        return W;
    }

    // Fixed-size eigendecomposition for small k — avoids all heap allocation
    // inside the Eigen solver. For k=15 this is ~3x faster than dynamic.
    template <int N>
    Eigen::Matrix<T, Eigen::Dynamic, 1> computeSpectralFixed(
            const Eigen::Matrix<T, N, N>& L, int k) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N>> es(L);
        int avail = std::min(k, N);
        Eigen::Matrix<T, Eigen::Dynamic, 1> features(k);
        for (int i = 0; i < avail; ++i) {
            features(i) = std::sqrt(std::abs(es.eigenvalues()(i))) *
                          es.eigenvectors().col(i)(0);
        }
        for (int i = avail; i < k; ++i) features(i) = T(0);
        return features;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> computeFastSpectralFeatures(const MatrixXT& L, int k) {
        const int n = static_cast<int>(L.rows());

        // Dispatch to fixed-size solver for common k values — stack-allocated,
        // no heap, fully unrolled by the compiler.
        if (n == 10) return computeSpectralFixed<10>(L, k);
        if (n == 15) return computeSpectralFixed<15>(L, k);
        if (n == 20) return computeSpectralFixed<20>(L, k);
        if (n == 30) return computeSpectralFixed<30>(L, k);

        if(n <= 32) {
            Eigen::SelfAdjointEigenSolver<MatrixXT> es(L);
            Eigen::Matrix<T, Eigen::Dynamic, 1> features(k);
            for(int i = 0; i < k; ++i) {
                features(i) = std::sqrt(std::abs(es.eigenvalues()(i))) *
                            es.eigenvectors().col(i)(0);
            }
            return features;
        }

        // Initialize with random vector
        VectorXT v = VectorXT::Random(n);
        v.normalize();

        // Storage for Arnoldi
        const int m = std::min(k + 5, n);
        MatrixXT V(n, m);
        MatrixXT H = MatrixXT::Zero(m, m);
        V.col(0) = v;

        // Modified Arnoldi iteration with reorthogonalization
        for(int j = 0; j < m - 1; ++j) {
            VectorXT w = L * V.col(j);

            // Modified Gram-Schmidt with reorthogonalization
            for(int i = 0; i <= j; ++i) {
                T h = w.dot(V.col(i));
                H(i,j) = h;
                w -= h * V.col(i);
            }

            T norm = w.norm();
            if(norm < 1e-10) break;

            H(j + 1,j) = norm;
            V.col(j + 1) = w / norm;
        }

        // Compute features from small eigendecomposition
        int h_size = std::max(1, m - 1);
        Eigen::SelfAdjointEigenSolver<MatrixXT> es(H.topLeftCorner(h_size, h_size));

        int effective_k = std::min(k, h_size);
        Eigen::Matrix<T, Eigen::Dynamic, 1> features = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(k);
        MatrixXT V_block = V.leftCols(h_size);
        for(int i = 0; i < effective_k; ++i) {
            features(i) = std::sqrt(std::abs(es.eigenvalues()(i))) *
                        (V_block * es.eigenvectors().col(i))(0);
        }

        return features;
    }

    // Isotropic Laplacian — fused weight + normalize in one pass.
    // For k <= 15 uses fixed-size matrices (stack allocated, no heap).
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLaplaceBeltrami(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud) {

        const int n = static_cast<int>(k_neighbors_);

        if (n == 15) return computeLaplaceBeltramiFixed<15>(local_cloud);
        if (n == 10) return computeLaplaceBeltramiFixed<10>(local_cloud);
        if (n == 20) return computeLaplaceBeltramiFixed<20>(local_cloud);
        if (n == 30) return computeLaplaceBeltramiFixed<30>(local_cloud);

        // Fallback for other sizes
        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        MatrixXT W = MatrixXT::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                T d2 = (local_cloud[i] - local_cloud[j]).squaredNorm();
                T w = computeWeight(d2);
                W(i, j) = w; W(j, i) = w;
            }
        }
        if (use_fast_approximation_) {
            return approximateLaplaceBeltrami(W, num_samples_);
        }
        VectorXT D = W.rowwise().sum();
        VectorXT Dinv = D.unaryExpr([](T v) -> T {
            return v > static_cast<T>(1e-10) ? static_cast<T>(1) / std::sqrt(v) : static_cast<T>(0);
        });
        // Fused: L_norm = Dinv * (D - W) * Dinv, done element-wise to avoid temporaries
        MatrixXT L(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T val = (i == j) ? (D(i) - W(i, j)) : -W(i, j);
                L(i, j) = Dinv(i) * val * Dinv(j);
            }
        }
        return L;
    }

    // Fixed-size version: entirely stack-allocated, no heap, compiler can unroll
    template <int N>
    Eigen::Matrix<T, N, N> computeLaplaceBeltramiFixed(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud) {

        Eigen::Matrix<T, N, N> W = Eigen::Matrix<T, N, N>::Zero();

        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                T d2 = (local_cloud[i] - local_cloud[j]).squaredNorm();
                T w = computeWeight(d2);
                W(i, j) = w; W(j, i) = w;
            }
        }

        if (use_fast_approximation_) {
            return approximateLaplaceBeltrami(MatrixXT(W), num_samples_);
        }

        // Fused normalized Laplacian — no temporaries, no heap
        Eigen::Matrix<T, N, 1> D = W.rowwise().sum();
        Eigen::Matrix<T, N, 1> Dinv;
        for (int i = 0; i < N; ++i)
            Dinv(i) = D(i) > static_cast<T>(1e-10) ? static_cast<T>(1) / std::sqrt(D(i)) : static_cast<T>(0);

        Eigen::Matrix<T, N, N> L;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                T val = (i == j) ? (D(i) - W(i, j)) : -W(i, j);
                L(i, j) = Dinv(i) * val * Dinv(j);
            }
        }
        return L;
    }

    // Anisotropic Laplacian (with normals) — fused weight + normalize
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLaplaceBeltrami(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud,
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_normals) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        const int n = static_cast<int>(k_neighbors_);

        MatrixXT W = computeEfficientWeights(local_cloud, local_normals);

        if (use_fast_approximation_) {
            return approximateLaplaceBeltrami(W, num_samples_);
        }

        // Fused normalized Laplacian
        VectorXT D = W.rowwise().sum();
        VectorXT Dinv = D.unaryExpr([](T v) -> T {
            return v > static_cast<T>(1e-10) ? static_cast<T>(1) / std::sqrt(v) : static_cast<T>(0);
        });
        MatrixXT L(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T val = (i == j) ? (D(i) - W(i, j)) : -W(i, j);
                L(i, j) = Dinv(i) * val * Dinv(j);
            }
        }
        return L;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> approximateLaplaceBeltrami(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& W, int num_samples) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        int n = W.rows();
        VectorXT D = W.rowwise().sum();

        // Sample unique indices for Nyström approximation
        std::vector<int> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        thread_local std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < num_samples; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            std::swap(pool[i], pool[dist(gen)]);
        }
        Eigen::VectorXi sample_indices(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            sample_indices[i] = pool[i];
        }
        MatrixXT W_sampled(n, num_samples);
        MatrixXT W_small(num_samples, num_samples);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                W_sampled(i, j) = W(i, sample_indices[j]);
            }
        }
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                W_small(i, j) = W(sample_indices[i], sample_indices[j]);
            }
        }
        VectorXT D_inv_sqrt = D.unaryExpr([](T v) -> T {
            return v > static_cast<T>(1e-10) ? static_cast<T>(1) / std::sqrt(v) : static_cast<T>(0);
        });
        W_sampled = D_inv_sqrt.asDiagonal() * W_sampled;
        for (int j = 0; j < num_samples; ++j) {
            W_sampled.col(j) *= D_inv_sqrt(sample_indices[j]);
        }
        for (int i = 0; i < num_samples; ++i) {
            W_small.row(i) *= D_inv_sqrt(sample_indices[i]);
            W_small.col(i) *= D_inv_sqrt(sample_indices[i]);
        }
        Eigen::SelfAdjointEigenSolver<MatrixXT> eigen_solver(W_small);
        MatrixXT U_small = eigen_solver.eigenvectors();
        VectorXT S_small = eigen_solver.eigenvalues();
        // Nyström approximation
        VectorXT S_small_inv = S_small.unaryExpr([](T v) -> T {
            return std::abs(v) > static_cast<T>(1e-10) ? static_cast<T>(1) / v : static_cast<T>(0);
        });
        MatrixXT U = W_sampled * U_small * S_small_inv.asDiagonal();
        // Orthogonalize U
        Eigen::HouseholderQR<MatrixXT> qr(U);
        U = qr.householderQ() * MatrixXT::Identity(n, num_samples);
        // Compute approximate Laplacian eigenvalues
        VectorXT ones = VectorXT::Ones(n);
        VectorXT S = (U.transpose() * (ones - W * U)).cwiseQuotient(U.colwise().sum());
        // Return approximated Laplacian
        return U * S.asDiagonal() * U.transpose();
    }

    static inline T approximateGeodesicDistance(const AlignedVector3<T>& p1, const AlignedVector3<T>& p2,
                                                const AlignedVector3<T>& n1, const AlignedVector3<T>& n2,
                                                T normal_weight) {
        T euclidean_dist = (p1 - p2).norm();
        T normal_diff = 1 - n1.dot(n2);
        // normal_weight controls how strongly a normal-orientation mismatch
        // stretches the approximate geodesic distance beyond the Euclidean one.
        return euclidean_dist * (1 + normal_weight * normal_diff);
    }

    static inline T computeAnisotropicWeight(const AlignedVector3<T>& p1, const AlignedVector3<T>& p2,
                                            const AlignedVector3<T>& n1, const AlignedVector3<T>& n2,
                                            T normal_weight) {
        T geo_dist = approximateGeodesicDistance(p1, p2, n1, n2, normal_weight);
        T anisotropic_factor = std::abs(n1.dot(p1 - p2));
        return anisotropic_factor / (1 + geo_dist);
    }

    static inline T computeWeight(T x) {
        if constexpr (std::is_same_v<T, float>) {
            if (x > 88.376f) return 0.0f;
            if (OPTIMAL_ALIGN >= 32) {
                __m256 vx = _mm256_set1_ps(-x);
                __m256 vresult = exp_ps_avx(vx);
                float result;
                _mm_store_ss(&result, _mm256_extractf128_ps(vresult, 0));
                return result;
            } else {
                __m128 vx = _mm_set_ss(-x);
                __m128 vresult = exp_ps(vx);
                float result;
                _mm_store_ss(&result, vresult);
                return result;
            }
        } else {
            return std::exp(-x);
        }
    }

    // SIMD exp() approximation — polynomial minimax for float precision.
    static inline __m128 exp_ps(__m128 x) {
        __m128 tmp = _mm_setzero_ps(), fx;
        __m128i emm0;
        __m128 one = _mm_set1_ps(1.0f);
        __m128 exp_hi = _mm_set1_ps(88.3762626647949f);
        __m128 exp_lo = _mm_set1_ps(-88.3762626647949f);

        x = _mm_min_ps(x, exp_hi);
        x = _mm_max_ps(x, exp_lo);

        fx = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f));

        emm0 = _mm_cvttps_epi32(fx);
        tmp = _mm_cvtepi32_ps(emm0);

        __m128 mask = _mm_cmpgt_ps(tmp, fx);
        mask = _mm_and_ps(mask, one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, _mm_set1_ps(-0.693359375f));
        __m128 z = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f));
        x = _mm_add_ps(x, tmp);
        x = _mm_add_ps(x, z);

        z = _mm_mul_ps(x, x);

        __m128 y = _mm_set1_ps(1.9875691500E-4f);
        y = _mm_mul_ps(y, x);
        y = _mm_add_ps(y, _mm_set1_ps(1.3981999507E-3f));
        y = _mm_mul_ps(y, x);
        y = _mm_add_ps(y, _mm_set1_ps(8.3334519073E-3f));
        y = _mm_mul_ps(y, x);
        y = _mm_add_ps(y, _mm_set1_ps(4.1665795894E-2f));
        y = _mm_mul_ps(y, x);
        y = _mm_add_ps(y, _mm_set1_ps(1.6666665459E-1f));
        y = _mm_mul_ps(y, x);
        y = _mm_add_ps(y, _mm_set1_ps(5.0000001201E-1f));
        y = _mm_mul_ps(y, z);
        y = _mm_add_ps(y, x);
        y = _mm_add_ps(y, one);

        emm0 = _mm_cvttps_epi32(fx);
        emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
        emm0 = _mm_slli_epi32(emm0, 23);
        __m128 pow2n = _mm_castsi128_ps(emm0);

        y = _mm_mul_ps(y, pow2n);
        return y;
    }

    static inline __m256 exp_ps_avx(__m256 x) {
        __m256 tmp = _mm256_setzero_ps(), fx;
        __m256i emm0;
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
        __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

        x = _mm256_min_ps(x, exp_hi);
        x = _mm256_max_ps(x, exp_lo);

        fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));

        emm0 = _mm256_cvttps_epi32(fx);
        tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
        mask = _mm256_and_ps(mask, one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_mul_ps(fx, _mm256_set1_ps(-0.693359375f));
        __m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
        x = _mm256_add_ps(x, tmp);
        x = _mm256_add_ps(x, z);

        z = _mm256_mul_ps(x, x);

        __m256 y = _mm256_set1_ps(1.9875691500E-4f);
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3f));
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(8.3334519073E-3f));
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(4.1665795894E-2f));
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.6666665459E-1f));
        y = _mm256_mul_ps(y, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(5.0000001201E-1f));
        y = _mm256_mul_ps(y, z);
        y = _mm256_add_ps(y, x);
        y = _mm256_add_ps(y, one);

        emm0 = _mm256_cvttps_epi32(fx);
        emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
        emm0 = _mm256_slli_epi32(emm0, 23);
        __m256 pow2n = _mm256_castsi256_ps(emm0);

        y = _mm256_mul_ps(y, pow2n);
        return y;
    }
};

} // namespace llb_features

#endif // LLB_FEATURES_HPP
