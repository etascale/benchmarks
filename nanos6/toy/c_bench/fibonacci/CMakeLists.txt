cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_c_benchmarks)

add_executable(fibonacci fibonacci.c)
target_link_libraries(fibonacci common)
target_link_libraries(fibonacci numa)
target_link_libraries(fibonacci rt)

install(TARGETS fibonacci DESTINATION bin)
