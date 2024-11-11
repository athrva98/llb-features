# LLB Features: Local Laplace-Beltrami Features for Point Clouds

## Introduction

LLB Features is a C++ library for computing Local Laplace-Beltrami (LLB) features on point clouds. The Laplace-Beltrami operator is a generalization of the Laplace operator to manifolds, providing a way to analyze the intrinsic geometry of surfaces represented as point clouds.

### What are Laplace-Beltrami Features?

Laplace-Beltrami features capture local and global geometric properties of a surface. They are:

- Invariant to isometric deformations (bending without stretching)
- Robust to noise and sampling density variations
- Capable of encoding multi-scale information about the surface

These properties make LLB features particularly useful in various point cloud processing tasks.

### Applications

1. **Point Cloud Registration**: LLB features can be used as descriptors for finding correspondences between different point clouds, aiding in alignment and registration tasks.

2. **Segmentation**: The features can help distinguish between different geometric regions, facilitating segmentation of point clouds into meaningful parts.

3. **Shape Classification**: LLB features provide a compact representation of local geometry, useful for training machine learning models for shape classification.

4. **Feature Detection**: They can be used to identify keypoints or interesting regions in a point cloud based on geometric properties.

5. **Shape Retrieval**: In large databases of 3D models, LLB features can be used to efficiently search for similar shapes.

6. **Surface Reconstruction**: The features can guide surface reconstruction algorithms by providing information about local geometry.

## Characteristics of this Implementation

- Computes Local Laplace-Beltrami features for 3D point clouds
- Utilizes Eigen for linear algebra operations
- Uses nanoflann for fast nearest neighbor searches
- OpenMP support for parallel computations
- AVX and SSE optimizations for improved performance

## Prerequisites

- CMake 3.12 or higher
- C++17 compatible compiler
- Eigen 3.3 or higher
- nanoflann 1.5.0 or higher
- OpenMP (optional, but recommended for better performance)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/athrva98/llb-features.git
cd llb-features
```
### 2. Install dependencies

Ensure you have Eigen and nanoflann installed on your system. You can download them from their respective websites:

- Eigen: http://eigen.tuxfamily.org/
- nanoflann: https://github.com/jlblancoc/nanoflann

Note the installation paths for both libraries.

### 3. Configure the project

Create a build directory and run CMake:

```bash
mkdir build
cd build
cmake ..
```

### 4. Build the project

```bash
cmake --build . --config Release
```

### 5. (Optional) Install the library

```bash
cmake --install . --config Release
```

## Usage

To use LLB Features in your project, include the `llb_features.hpp` header and link against Eigen and nanoflann.

Here's a simple example:

```cpp
#include <llb_features.hpp>
#include <vector>
#include <iostream>

// Utility function to convert std::vector<Eigen::Vector3f> to LLB Features format
template<typename T>
llb_features::AlignedVector<T> convertStdEigenVectorToLLB(const std::vector<Eigen::Vector3f>& cloud) {
    llb_features::AlignedVector<T> llb_points;
    llb_points.reserve(cloud.size());
    for (const auto& point : cloud) {
        llb_points.emplace_back(point.template cast<T>());
    }
    return llb_points;
}


int main() {
    try {
        // Create a simple point cloud
        std::vector<Eigen::Vector3f> point_cloud = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {1.0f, 1.0f, 0.0f},
            {1.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 1.0f}
        };

        // Create an instance of LLBFeatures with safer parameters
        llb_features::LLBFeatures<float> llb(
            convertStdEigenVectorToLLB<float>(point_cloud),
            3,  // k_neighbors
            2   // num_eigenvectors
        );

        // Compute the features with error handling
        std::vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> features;
        try {
            features = llb.computeFeatures();
        } catch (const std::exception& e) {
            std::cerr << "Error computing features: " << e.what() << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### Usage with Open3D

LLB Features can be easily integrated with Open3D point clouds. Here's an example of how to use LLB Features with an Open3D point cloud.
LLB Features can also use the point normals of the PointClouds if they are available.
In general, always prefer the variant that uses normals (i.e., compute normals for point clouds). **The variant that does not use normals will be removed in the future.**

```cpp
// Example with normal information
#include <iostream>
#include <open3d/Open3D.h>
#include <llb_features.hpp>

using PointCloud = open3d::geometry::PointCloud;
using Feature = open3d::pipelines::registration::Feature;

template<typename T>
std::pair<llb_features::AlignedVector<T>, llb_features::AlignedVector<T>>
convertOpen3DToLLB(const PointCloud& cloud, bool hasNormals) {
    llb_features::AlignedVector<T> llb_points;
    llb_features::AlignedVector<T> llb_normals;

    llb_points.reserve(cloud.points_.size());
    for (const auto& point : cloud.points_) {
        llb_points.emplace_back(point.cast<T>());
    }

    if (hasNormals) {
        llb_normals.reserve(cloud.normals_.size());
        for (const auto& normal : cloud.normals_) {
            llb_normals.emplace_back(normal.cast<T>());
        }
    }

    return {llb_points, llb_normals};
}

template<typename T>
Feature ConvertLLBToOpen3DFeatures(
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>,
    Eigen::aligned_allocator<Eigen::Matrix<T, Eigen::Dynamic, 1>>>& llb_features) {
    int feature_dim = llb_features.front().rows();
    int num_features = llb_features.size();

    Feature open3d_features;
    open3d_features.Resize(feature_dim, num_features);

    for (int i = 0; i < num_features; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            open3d_features.data_(j, i) = llb_features[i](j);
        }
    }
    return open3d_features;
}

int main() {
    // Load point cloud with normals
    std::string cloud_path = "path_to_pointcloud_with_normals.ply";
    auto cloud = open3d::io::CreatePointCloudFromFile(cloud_path);

    if (!cloud || !cloud->HasNormals()) {
        std::cerr << "Failed to load point cloud with normals" << std::endl;
        return 1;
    }

    // Convert to LLB format
    auto [points, normals] = convertOpen3DToLLB<float>(*cloud, true);

    // Compute LLB features
    llb_features::LLBFeatures<float> llb(points, normals, 155, 15, 0.45);
    auto features = llb.computeFeatures();

    // Convert LLB features to Open3D features
    Feature o3d_features = ConvertLLBToOpen3DFeatures<float>(features);

    std::cout << "LLB features computed successfully with normals." << std::endl;
    std::cout << "Number of features: " << features.size() << std::endl;
    std::cout << "Feature dimension: " << features[0].rows() << std::endl;

    return 0;
}
```

Here is an example that does not use the normal information:

```cpp
// Example without normal information
#include <iostream>
#include <open3d/Open3D.h>
#include <llb_features.hpp>

using PointCloud = open3d::geometry::PointCloud;
using Feature = open3d::pipelines::registration::Feature;

template<typename T>
std::pair<llb_features::AlignedVector<T>, llb_features::AlignedVector<T>>
convertOpen3DToLLB(const PointCloud& cloud, bool hasNormals) {
    llb_features::AlignedVector<T> llb_points;
    llb_features::AlignedVector<T> llb_normals;

    llb_points.reserve(cloud.points_.size());
    for (const auto& point : cloud.points_) {
        llb_points.emplace_back(point.cast<T>());
    }

    if (hasNormals) {
        llb_normals.reserve(cloud.normals_.size());
        for (const auto& normal : cloud.normals_) {
            llb_normals.emplace_back(normal.cast<T>());
        }
    }

    return {llb_points, llb_normals};
}

template<typename T>
Feature ConvertLLBToOpen3DFeatures(
    const std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>,
    Eigen::aligned_allocator<Eigen::Matrix<T, Eigen::Dynamic, 1>>>& llb_features) {
    int feature_dim = llb_features.front().rows();
    int num_features = llb_features.size();

    Feature open3d_features;
    open3d_features.Resize(feature_dim, num_features);

    for (int i = 0; i < num_features; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            open3d_features.data_(j, i) = llb_features[i](j);
        }
    }
    return open3d_features;
}

int main() {
    // Load point cloud without normals
    std::string cloud_path = "path_to_pointcloud_without_normals.ply";
    auto cloud = open3d::io::CreatePointCloudFromFile(cloud_path);

    if (!cloud) {
        std::cerr << "Failed to load point cloud" << std::endl;
        return 1;
    }

    // Convert to LLB format
    auto points = convertOpen3DToLLB<float>(*cloud);

    // Compute LLB features
    llb_features::LLBFeatures<float> llb(points, 155, 15);
    auto features = llb.computeFeatures();

    // Convert LLB features to Open3D features
    Feature o3d_features = ConvertLLBToOpen3DFeatures<float>(features);

    std::cout << "LLB features computed successfully without normals." << std::endl;
    std::cout << "Number of features: " << features.size() << std::endl;
    std::cout << "Feature dimension: " << features[0].rows() << std::endl;

    return 0;
}
```

## CMake Integration

If you've installed the library, you can use it in your CMake project:

```cmake
cmake_minimum_required(VERSION 3.12)
project(your_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLBFeatures REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(your_target main.cpp)
target_link_libraries(your_target PRIVATE
    LLBFeatures::llb_features
    OpenMP::OpenMP_CXX
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Eigen](http://eigen.tuxfamily.org/) library for linear algebra
- [nanoflann](https://github.com/jlblancoc/nanoflann) library for fast nearest neighbor searches

## Citation

If you use LLB Features in your research or project, please cite it as follows:

```bibtex
@software{llb_features,
  author = {Pandhare, Athrva},
  title = {LLB Features: Local Laplace-Beltrami Features Library for Point Clouds},
  year = {2024},
  url = {https://github.com/athrva98/llb-features},
}
```

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.
