cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_cpp_benchmarks)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --ompss-2")

# nanos6 allocation wrappers
include_directories(common)

# available benchmarks
add_subdirectory(himeno)
add_subdirectory(matmul)
add_subdirectory(stream)
