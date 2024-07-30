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
- C++14 compatible compiler
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
cmake .. -DEIGEN3_CMAKE_DIR=/path/to/eigen/cmake -DNANOFLANN_CMAKE_DIR=/path/to/nanoflann/cmake
```

Replace `/path/to/eigen/cmake` and `/path/to/nanoflann/cmake` with the actual paths to the CMake configuration directories for Eigen and nanoflann on your system.

For example:

```bash
cmake .. -DEIGEN3_CMAKE_DIR="C:\Libraries\eigen-3.4.0\share\eigen3\cmake" -DNANOFLANN_CMAKE_DIR="C:\Libraries\nanoflann-1.5.0\share\nanoflann\cmake"
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

int main() {
    // Create a simple point cloud (replace this with your actual point cloud data)
    std::vector<Eigen::Vector3f> point_cloud = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    // Create an instance of LLBFeatures
    llb_features::LLBFeatures<float> llb(point_cloud);

    // Compute the features
    auto features = llb.computeFeatures();

    // Print the features
    for (size_t i = 0; i < features.size(); ++i) {
        std::cout << "Features for point " << i << ":\n" << features[i] << "\n\n";
    }

    return 0;
}
```

### Usage with Open3D

LLB Features can be easily integrated with Open3D point clouds. Here's an example of how to use LLB Features with an Open3D point cloud:

```cpp
#include <open3d/Open3D.h>
#include <llb_features.hpp>
#include <vector>
#include <iostream>

// Utility function to convert Open3D point cloud to LLB Features format
template<typename T>
std::vector<Eigen::Vector3<T>> convertOpen3DToLLB(const open3d::geometry::PointCloud& point_cloud) {
    std::vector<Eigen::Vector3<T>> llb_points;
    llb_points.reserve(point_cloud.points_.size());
    for (const auto& point : point_cloud.points_) {
        llb_points.emplace_back(point.cast<T>());
    }
    return llb_points;
}

int main() {
    // Load a point cloud using Open3D
    auto pcd = open3d::io::CreatePointCloudFromFile("path/to/your/pointcloud.ply");

    // Convert Open3D point cloud to LLB Features format
    auto llb_points = convertOpen3DToLLB<float>(*pcd);

    // Create an instance of LLBFeatures
    llb_features::LLBFeatures<float> llb(llb_points);

    // Compute the features
    auto features = llb.computeFeatures();

    // Print the first few features
    std::cout << "Features for the first 5 points:\n";
    for (size_t i = 0; i < std::min(size_t(5), features.size()); ++i) {
        std::cout << "Point " << i << ":\n" << features[i].transpose() << "\n\n";
    }

    return 0;
}
```

## CMake Integration

If you've installed the library, you can use it in your CMake project like this:

```cmake
find_package(LLBFeatures REQUIRED)
target_link_libraries(your_target PRIVATE LLBFeatures::llb_features)
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
