cmake_minimum_required(VERSION 3.12)
project(LLBFeatures VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)

# Find Eigen3 from the workspace
find_package(Eigen3 3.3 REQUIRED)

# Find nanoflann from the workspace
find_package(nanoflann REQUIRED)

# Create the library
add_library(llb_features INTERFACE)
target_include_directories(llb_features INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(llb_features INTERFACE Eigen3::Eigen nanoflann::nanoflann)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llb_features INTERFACE OpenMP::OpenMP_CXX)
endif()

# Enable AVX and SSE instructions
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(llb_features INTERFACE -mavx2)
endif()
CHECK_CXX_COMPILER_FLAG("-msse4.2" COMPILER_SUPPORTS_SSE4_2)
if(COMPILER_SUPPORTS_SSE4_2)
    target_compile_options(llb_features INTERFACE -msse4.2)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS llb_features
    EXPORT LLBFeaturesTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES llb_features.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT LLBFeaturesTargets
    FILE LLBFeaturesTargets.cmake
    NAMESPACE LLBFeatures::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLBFeatures
)

# Create and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/LLBFeaturesConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLBFeatures
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LLBFeaturesConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLBFeatures
)
