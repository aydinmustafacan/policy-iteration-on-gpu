cmake_minimum_required(VERSION 3.29)
project(grid_world_generation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize Release builds
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Include FetchContent for GoogleTest
include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---- Main executable ----
add_executable(grid_world_generation
        main.cc
        policy_iteration.cc
        policy_improvement_sparse.cc
        grid_gen.cc
        io.cc
        controlled.cc
)

target_compile_features(grid_world_generation PRIVATE cxx_std_20)

# ---- Tests ----
enable_testing()

# Create test executable for grid world functionality
add_executable(grid_world_tests
        tests/grid_world_test.cpp
        tests/grid_world_format_test.cpp
        grid_gen.cc
        io.cc
        controlled.cc
        policy_iteration.cc
        policy_improvement_sparse.cc
)

# Link libraries for tests
target_link_libraries(grid_world_tests
        gtest_main
        gtest
)

# Include directories for tests
target_include_directories(grid_world_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(grid_world_tests)

# Original tests (if they exist)
add_executable(tests
        policy_iteration.cc
        policy_improvement_sparse.cc
)

target_link_libraries(tests PRIVATE gtest_main)
gtest_discover_tests(tests)
