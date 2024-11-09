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
#include <vector>
#include <memory>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <nanoflann.hpp>

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
        desc.variation = eigenvalues(0) / sum_eigenvalues;
        desc.planarity = (eigenvalues(1) - eigenvalues(0)) / eigenvalues(2);
        desc.anisotropy = (eigenvalues(2) - eigenvalues(0)) / eigenvalues(2);
        desc.sphericity = eigenvalues(0) / eigenvalues(2);
        desc.principal_dir = eigenvectors.col(2);

        // Compute stability score based on eigenvalue ratios
        T e1_ratio = eigenvalues(1) / eigenvalues(2);
        T e0_ratio = eigenvalues(0) / eigenvalues(1);
        desc.stability_score = std::abs(e1_ratio - e0_ratio) / (1 + e1_ratio + e0_ratio);

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

        // Compute median distance for robust scaling
        size_t mid = distances.size() / 2;
        std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
        T median_dist = distances[mid];
        T huber_threshold = 1.345 * median_dist; // Huber's robust estimator constant

        // Second pass: compute weighted covariance
        T total_weight = 0;
        for (size_t i = 0; i < neighborhood.size(); ++i) {
            AlignedVector3<T> centered = neighborhood[i] - query;
            T dist = distances[i];

            // Huber weight
            T weight = (dist < huber_threshold) ?
                      1.0 : (huber_threshold / dist);

            covariance += weight * (centered * centered.transpose());
            total_weight += weight;
        }

        return covariance / total_weight;
    }

private:
    static constexpr T epsilon = 1e-10;
    static constexpr size_t MIN_NEIGHBORS = 5;

};

template <typename T>
class alignas(OPTIMAL_ALIGN) LLBPointCloud {
public:
    AlignedVector<T> points;

    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <typename T>
class alignas(OPTIMAL_ALIGN) LLBFeatures {
    using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
public:
    LLBFeatures(const AlignedVector<T>& point_cloud,
                size_t k_neighbors = 20,
                size_t num_eigenvectors = 10,
                bool use_fast_approximation = false,
                size_t num_samples = 100)
        : point_cloud_(point_cloud),
          k_neighbors_(k_neighbors),
          num_eigenvectors_(num_eigenvectors),
          use_fast_approximation_(use_fast_approximation),
          num_samples_(num_samples) {
        validateInputs();
        buildKDTree();
    }

    LLBFeatures(const AlignedVector<T>& point_cloud,
                const AlignedVector<T>& normals,
                size_t k_neighbors = 20,
                size_t num_eigenvectors = 10,
                const T normal_weight_factor = 0.1,
                bool use_fast_approximation = false,
                size_t num_samples = 100)
        : point_cloud_(point_cloud),
          normals_(normals),
          k_neighbors_(k_neighbors),
          num_eigenvectors_(num_eigenvectors),
          normal_weight_factor_(normal_weight_factor),
          use_fast_approximation_(use_fast_approximation),
          num_samples_(num_samples) {
        validateInputs();
        buildKDTree();
    }

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> computeFeatures() {
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> features(point_cloud_.size());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(point_cloud_.size()); ++i) {
            features[i] = computeLocalFeatures(i);
        }

        return features;
    }

private:
    const AlignedVector<T>& point_cloud_;
    const AlignedVector<T>& normals_ = {};
    size_t k_neighbors_;
    T normal_weight_factor_;
    size_t num_eigenvectors_;
    bool use_fast_approximation_;
    size_t num_samples_;
    LLBPointCloud<T> pc_adaptor_;
    std::unique_ptr<nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L1_Adaptor<T,LLBPointCloud<T>>, LLBPointCloud<T>, 3>> kdtree_;

    void validateInputs() {
        if (point_cloud_.empty()) {
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
    }

    void buildKDTree() {
        pc_adaptor_.points = point_cloud_;
        kdtree_ = std::make_unique<nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L1_Adaptor<T, LLBPointCloud<T>>, LLBPointCloud<T>, 3>>(
            3, pc_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree_->buildIndex();
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> computeLocalFeatures(size_t point_idx) {
        std::vector<size_t> neighbor_indices(k_neighbors_);
        std::vector<T> distances(k_neighbors_);

        nanoflann::KNNResultSet<T> resultSet(k_neighbors_);
        resultSet.init(neighbor_indices.data(), distances.data());

        kdtree_->findNeighbors(resultSet, point_cloud_[point_idx].data(), nanoflann::SearchParameters());

        std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>> local_cloud(k_neighbors_);
        std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>> local_normals;

        AlignedVector3<T> center = {0, 0, 0};
        for (size_t i = 0; i < k_neighbors_; ++i) {
            local_cloud[i] = point_cloud_[neighbor_indices[i]];
            center += local_cloud[i];
        }
        center /= local_cloud.size();

        Eigen::Matrix<T, 3, 3> covariance = LocalGeometryAnalyzer<T>::computeRobustCovariance(center, local_cloud);

        LocalGeometryAnalyzer<T>::LocalSurfaceDescriptor desc = LocalGeometryAnalyzer<T>::computeLocalSurfaceProperties(center, local_cloud);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> L;

        if (!normals_.empty()) {
            local_normals.resize(k_neighbors_);
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

        // Compute local scale using k-nearest neighbors
        // Use at most 20 neighbors for scale
        int k = std::min<int>(20, k_neighbors_);
        std::vector<T> knn_distances(k_neighbors_);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < k_neighbors_; ++i) {
            std::vector<T> dists;
            dists.reserve(k_neighbors_);
            for (int j = 0; j < k_neighbors_; ++j) {
                if (i != j) {
                    dists.push_back((local_cloud[i] - local_cloud[j]).squaredNorm());
                }
            }
            std::nth_element(dists.begin(), dists.begin() + k - 1, dists.end());
            knn_distances[i] = std::sqrt(dists[k - 1]);
        }

        // Use median of k-nn distances for scale
        std::nth_element(knn_distances.begin(),
                        knn_distances.begin() + k_neighbors_/2,
                        knn_distances.end());
        T scale = knn_distances[k_neighbors_/2];
        T squared_scale = scale * scale;

        constexpr int BLOCK_SIZE = 32;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < k_neighbors_; ++i) {
            for (int j = i + 1; j < k_neighbors_; j += BLOCK_SIZE) {
                int block_end = std::min<int>(j + BLOCK_SIZE, k_neighbors_);

                // Compute weights for block with anisotropic weighting
                for (int k = 0; k < block_end - j; ++k) {
                    const auto& p1 = local_cloud[i];
                    const auto& p2 = local_cloud[j + k];
                    const auto& n1 = local_normals[i];
                    const auto& n2 = local_normals[j + k];

                    T dist = (p1 - p2).squaredNorm();
                    if (dist < squared_scale * static_cast<T>(9.0)) {
                        T weight = computeAnisotropicWeight(
                            p1, p2, n1, n2);
                        W(i, j + k) = W(j + k, i) = weight;
                    }
                }
            }
        }
        return W;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> computeFastSpectralFeatures(const MatrixXT& L, int k) {
        const int n = L.rows();
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
        Eigen::SelfAdjointEigenSolver<MatrixXT> es(H.topLeftCorner(m-1, m-1));

        Eigen::Matrix<T, Eigen::Dynamic, 1> features(k);
        for(int i = 0; i < k; ++i) {
            features(i) = std::sqrt(std::abs(es.eigenvalues()(i))) *
                        (V.leftCols(m-1) * es.eigenvectors().col(i))(0);
        }

        return features;
    }

    // variant without normal weighing
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLaplaceBeltrami(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        MatrixXT W = MatrixXT::Zero(k_neighbors_, k_neighbors_);

        for (int i = 0; i < static_cast<int>(k_neighbors_); ++i) {
            for (int j = i + 1; j < static_cast<int>(k_neighbors_); ++j) {
                T dist = (local_cloud[i] - local_cloud[j]).squaredNorm();
                T weight = computeWeight(dist);
                W(i, j) = W(j, i) = weight;
            }
        }

        if (use_fast_approximation_) {
            // nystrom approximation for computation
            return approximateLaplaceBeltrami(W, num_samples_);
        }
        else {
            VectorXT D = W.rowwise().sum();
            MatrixXT D_diag = D.asDiagonal();
            MatrixXT L = D_diag - W;
            MatrixXT D_inv_sqrt = D.cwiseInverse().cwiseSqrt().asDiagonal();
            return D_inv_sqrt * L * D_inv_sqrt;
        }
    }

    // variant with normal weighing
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLaplaceBeltrami(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud,
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_normals) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        MatrixXT W = computeEfficientWeights(local_cloud, local_normals);
        MatrixXT result;
        if (use_fast_approximation_) {
            result = approximateLaplaceBeltrami(W, num_samples_);
        }
        else {
            VectorXT D = W.rowwise().sum();
            MatrixXT D_diag = D.asDiagonal();
            MatrixXT L = D_diag - W;
            MatrixXT D_inv_sqrt = D.cwiseInverse().cwiseSqrt().asDiagonal();
            result = D_inv_sqrt * L * D_inv_sqrt;
        }

        return result;
    }

    // TODO: Test and optimize this
    // void computeMultiScaleFeatures(
    //     const std::vector<size_t>& scales,
    //     std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>>& multi_scale_features) {
    //     multi_scale_features.resize(scales.size());
    //     // Estimate a good cache block size
    //     constexpr size_t typical_l1_cache_size = 32 * 1024;
    //     constexpr size_t point_size = sizeof(T) * 3;
    //     constexpr size_t cache_block_size = typical_l1_cache_size / point_size;
    //     const size_t effective_block_size = std::max<size_t>(1, std::min<size_t>(cache_block_size, point_cloud_.size()));

    //     #pragma omp parallel for schedule(dynamic)
    //     for (int i = 0; i < static_cast<int>(scales.size()); ++i) {  // Fixed the loop condition
    //         size_t current_scale = scales[i];
    //         multi_scale_features[i].resize(point_cloud_.size());
    //         // Process points in a cache-friendly manner
    //         for (size_t start = 0; start < point_cloud_.size(); start += effective_block_size) {
    //             size_t end = std::min<size_t>(start + effective_block_size, point_cloud_.size());
    //             computeFeaturesForBlock(start, end, current_scale, multi_scale_features[i]);
    //         }
    //     }
    // }

    // void computeFeaturesForBlock(
    //     const size_t start,
    //     const size_t end,
    //     const size_t scale,
    //     std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>& features) {
    //     const size_t original_k_neighbors = k_neighbors_;
    //     k_neighbors_ = scale;
    //     for (size_t j = start; j < end; ++j) {
    //         features[j] = computeLocalFeatures(j);
    //     }
    //     k_neighbors_ = original_k_neighbors;
    // }

    T computeCurvatureFeatures(const Eigen::Matrix<T, 3, 3>& covariance_matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigen_solver;
        eigen_solver.computeDirect(covariance_matrix);

        Eigen::Matrix<T, 3, 1> eigenvalues = eigen_solver.eigenvalues();

        if (eigenvalues[0] > eigenvalues[1]) std::swap(eigenvalues[0], eigenvalues[1]);
        if (eigenvalues[1] > eigenvalues[2]) std::swap(eigenvalues[1], eigenvalues[2]);
        if (eigenvalues[0] > eigenvalues[1]) std::swap(eigenvalues[0], eigenvalues[1]);

        T sum_eigenvalues = eigenvalues.sum();
        if (sum_eigenvalues == 0) {
            return 0;
        }
        T mean_curvature = eigenvalues[1] / sum_eigenvalues;
        T gaussian_curvature = eigenvalues.prod() / (sum_eigenvalues * sum_eigenvalues);
        Eigen::Matrix<T, 2, 1> result;
        if constexpr (std::is_same_v<T, float> && OPTIMAL_ALIGN >= 16) {
            __m128 mean_gauss = _mm_set_ps(0, 0, gaussian_curvature, mean_curvature);
            _mm_store_ps(result.data(), mean_gauss);
        } else if constexpr (std::is_same_v<T, double> && OPTIMAL_ALIGN >= 32) {
            __m256d mean_gauss = _mm256_set_pd(0, 0, gaussian_curvature, mean_curvature);
            _mm256_store_pd(result.data(), mean_gauss);
        } else {
            result << mean_curvature, gaussian_curvature;
        }
        return static_cast<T>(result.squaredNorm());
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> approximateLaplaceBeltrami(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& W, int num_samples) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        int n = W.rows();
        VectorXT D = W.rowwise().sum();

        // Sample indices for Nyström approximation
        Eigen::VectorXi sample_indices(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            sample_indices[i] = rand() % n;
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
        VectorXT D_inv_sqrt = D.cwiseInverse().cwiseSqrt();
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
        MatrixXT U = W_sampled * U_small * S_small.cwiseInverse().asDiagonal();
        // Orthogonalize U
        Eigen::HouseholderQR<MatrixXT> qr(U);
        U = qr.householderQ() * MatrixXT::Identity(n, num_samples);
        // Compute approximate Laplacian eigenvalues
        VectorXT ones = VectorXT::Ones(n);
        VectorXT S = (U.transpose() * (ones - W * U)).cwiseQuotient(U.colwise().sum());
        // Return approximated Laplacian
        return U * S.asDiagonal() * U.transpose();
    }

    static inline T approximateGeodesicDistance(const AlignedVector3<T>& p1, const AlignedVector3<T>& p2, const AlignedVector3<T>& n1, const AlignedVector3<T>& n2) {
        T euclidean_dist = (p1 - p2).norm();
        T normal_diff = 1 - n1.dot(n2);
        return euclidean_dist * (1 + normal_diff);
    }

    static inline T computeAnisotropicWeight(const AlignedVector3<T>& p1, const AlignedVector3<T>& p2,
                                            const AlignedVector3<T>& n1, const AlignedVector3<T>& n2) {
        T geo_dist = approximateGeodesicDistance(p1, p2, n1, n2);
        T anisotropic_factor = std::abs(n1.dot(p1 - p2));
        return anisotropic_factor  / (1 + geo_dist);
    }

    static inline T computeWeight(T x, T ndiff, const double n_w) {
        T combined_dist = x + ndiff * n_w;
        if (OPTIMAL_ALIGN >= 32) {
            return computeWeightAVX(combined_dist);
        } else {
            return computeWeightSSE(combined_dist);
        }
    }

    static inline T computeWeight(T x) {
        if (OPTIMAL_ALIGN >= 32) {
            return computeWeightAVX(x);
        } else {
            return computeWeightSSE(x);
        }
    }

    static inline T computeWeightSSE(T x) {
        __m128 vx = _mm_set_ss(-x);
        __m128 vresult = exp_ps(vx);
        T result;
        _mm_store_ss(&result, vresult);
        return result;
    }

    static inline T computeWeightAVX(T x) {
        __m256 vx = _mm256_set1_ps(-x);
        __m256 vresult = exp_ps_avx(vx);
        T result;
        _mm_store_ss(&result, _mm256_extractf128_ps(vresult, 0));
        return result;
    }

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
