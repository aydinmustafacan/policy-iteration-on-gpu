cmake_minimum_required(VERSION 3.26)
project(serial)

set(CMAKE_CXX_STANDARD 20)

# Include FetchContent module
include(FetchContent)

# Fetch Google Test - using latest stable release
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Main executable (remove dense policy_iteration.cc)
add_executable(
        serial 
        main.cc
        writer.cc
        json_reader.cc
        sparse_policy_iteration.cc
)

# Test executable: remove dense policy tests and sources
add_executable(
        tests
        writer.cc
        json_reader.cc
        sparse_policy_iteration.cc
)

# Link Google Test to the test executable
target_link_libraries(
        tests
        gtest_main
)

# Enable testing
include(GoogleTest)
gtest_discover_tests(tests)