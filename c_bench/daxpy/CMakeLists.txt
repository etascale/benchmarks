cmake_minimum_required(VERSION 2.8)

project(ompss_cluster_c_benchmarks)

add_executable(daxpy-strong daxpy-strong.c)
target_link_libraries(daxpy-strong common)
target_link_libraries(daxpy-strong numa)
target_link_libraries(daxpy-strong rt)

add_executable(daxpy-weak daxpy-weak.c)
target_link_libraries(daxpy-weak common)
target_link_libraries(daxpy-weak numa)
target_link_libraries(daxpy-weak rt)

install(TARGETS daxpy-strong daxpy-weak DESTINATION bin)
