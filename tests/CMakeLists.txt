cmake_minimum_required(VERSION 3.12)
project(LLBFeaturesTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(OpenMP)

# Catch2 single-header, vendored
add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Unit tests
add_executable(llb_tests
    test_main.cpp
    test_local_geometry.cpp
    test_llb_features.cpp
    test_validation.cpp
    test_spectral.cpp
    test_weights.cpp
    test_regression.cpp
)
target_include_directories(llb_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(llb_tests PRIVATE Catch2 Eigen3::Eigen nanoflann::nanoflann)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llb_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(llb_tests PRIVATE /W3)
    target_compile_definitions(llb_tests PRIVATE _USE_MATH_DEFINES)
endif()

# Performance micro-benchmark
add_executable(perf_micro perf_micro.cpp)
target_include_directories(perf_micro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(perf_micro PRIVATE Eigen3::Eigen nanoflann::nanoflann)
if(OpenMP_CXX_FOUND)
    target_link_libraries(perf_micro PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(perf_micro PRIVATE /W3 /O2 /arch:AVX2 /fp:fast /Zi)
    target_link_options(perf_micro PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    target_compile_definitions(perf_micro PRIVATE _USE_MATH_DEFINES)
endif()

# 3DMatch evaluation harness (requires data/ directory with 3DMatch scenes)
add_executable(proper_eval proper_eval.cpp)
target_include_directories(proper_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(proper_eval PRIVATE Eigen3::Eigen nanoflann::nanoflann)
if(OpenMP_CXX_FOUND)
    target_link_libraries(proper_eval PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(proper_eval PRIVATE /W3 /O2 /arch:AVX2 /fp:fast)
    target_compile_definitions(proper_eval PRIVATE _USE_MATH_DEFINES)
endif()

enable_testing()
add_test(NAME llb_tests COMMAND llb_tests)
