cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler path
set(CMAKE_CUDA_COMPILER "/opt/apps/pkg/compilers/cuda/12.6.2/bin/nvcc")

# Project name and languages
project(cuda2 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard (optional, inherits from CXX if not set)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Create executable
add_executable(cuda2 main.cu json_reader.cpp)

# Set properties for CUDA compilation
set_target_properties(cuda2 PROPERTIES
    CUDA_ARCHITECTURES "89"  # Adjust for your GPU architecture
)

# Optional: build with single precision (float) on GPU
option(USE_FP32 "Build CUDA solver in single precision (float)" OFF)
if (USE_FP32)
    target_compile_definitions(cuda2 PRIVATE USE_FP32=1)
endif()