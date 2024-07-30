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
public:
    LLBFeatures(const AlignedVector<T>& point_cloud,
                size_t k_neighbors = 20,
                size_t num_eigenvectors = 10)
        : point_cloud_(point_cloud),
          k_neighbors_(k_neighbors),
          num_eigenvectors_(num_eigenvectors) {
        validateInputs();
        buildKDTree();
    }

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> computeFeatures() {
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> features(point_cloud_.size());

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(point_cloud_.size()); ++i) {
            features[i] = computeLocalFeatures(i);
        }

        return features;
    }

private:
    const AlignedVector<T>& point_cloud_;
    size_t k_neighbors_;
    size_t num_eigenvectors_;
    LLBPointCloud<T> pc_adaptor_;
    std::unique_ptr<nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T,LLBPointCloud<T>>, LLBPointCloud<T>, 3>> kdtree_;

    void validateInputs() {
        if (point_cloud_.empty()) {
            throw std::invalid_argument("Point cloud is empty");
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
            nanoflann::L2_Simple_Adaptor<T, LLBPointCloud<T>>, LLBPointCloud<T>, 3>>(
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
        for (size_t i = 0; i < k_neighbors_; ++i) {
            local_cloud[i] = point_cloud_[neighbor_indices[i]];
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> L = computeLaplaceBeltrami(local_cloud);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigensolver(L);
        return eigensolver.eigenvectors().col(0).head(num_eigenvectors_);
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> computeLaplaceBeltrami(
        const std::vector<AlignedVector3<T>, Eigen::aligned_allocator<AlignedVector3<T>>>& local_cloud) {

        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        MatrixXT W = MatrixXT::Zero(k_neighbors_, k_neighbors_);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(k_neighbors_); ++i) {
            for (int j = i + 1; j < static_cast<int>(k_neighbors_); ++j) {
                T dist = (local_cloud[i] - local_cloud[j]).squaredNorm();
                T weight = computeWeight(dist);
                W(i, j) = W(j, i) = weight;
            }
        }

        VectorXT D = W.rowwise().sum();
        MatrixXT D_diag = D.asDiagonal();

        MatrixXT L = D_diag - W;

        MatrixXT D_inv_sqrt = D.cwiseInverse().cwiseSqrt().asDiagonal();

        return D_inv_sqrt * L * D_inv_sqrt;
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
